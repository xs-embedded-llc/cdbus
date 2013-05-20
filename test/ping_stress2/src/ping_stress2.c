/*
 ============================================================================
 Name        : ping_stress2.c
 Author      : 
 Version     :
 Description : CDbus Ping Stress Test App
 ============================================================================
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <getopt.h>
#include <signal.h>
#include <assert.h>
#include <sys/stat.h>
#include <unistd.h>
#include <errno.h>
#include <fcntl.h>
#include <pthread.h>
#include <semaphore.h>
#include "cdbus/cdbus.h"
#include "dbus/dbus.h"
#include "ev.h"

/*---------------
  -- Const
  ---------------*/
#define APP_VER                 "1.0"
#define STRESS_BUSNAME          "com.dbus.service.ping_stress"
#define STRESS_OBJPATH          "/com/dbus/service/ping_stress"
#define STRESS_INTERFACE        "com.dbus.ping"
#define STRESS_ERRORIFACE       "com.dbus.ping.Error"

#define ASYNC_TIMEOUT           250  /* msec for server to reply on async call */
#define ASYNC_REPLY_TIMEOUT     2000 /* msec for client to wait for reply */

#define HELP_TEXT               "                                                          \n\
  This tool is used to stress the l2dbus interface using the shim API.  This               \n\
  tool implements a simple ping protocol which is configurable for various                 \n\
  testing scenarios.                                                                       \n\
                                                                                           \n\
  Usage:                                                                                   \n\
  Run one side as the service side using '-S', then run one or more clients                \n\
  without the '-S' option.                                                                 \n\
                                                                                           \n\
  Command line arguments:                                                                  \n\
                                                                                           \n\
  Common Args:                                                                             \n\
  --bus [busName]       -- Bus name                                                        \n\
  -v                    -- Verbosity                                                       \n\
                           v = level 1, Basic Count                                        \n\
                               Shows the packet count only                                 \n\
                           vv = level 2, Basic I/O                                         \n\
                               client: 't' == tx (request)                                 \n\
                                       'r' == rx on reply interface (reply)                \n\
                                       'e' == rx on error interface (error)                \n\
                                       '.' == wait                                         \n\
                               server: '.' == rx/tx on echo interface                      \n\
                                       'e' == rx/tx on echo as error interface             \n\
                                       'r'/'t' == rx/tx on echo async interface            \n\
                                       'R'/'T' == rx/tx on echo async as error interface   \n\
                           vvv = level 3, verbose output                                   \n\
                               Displays the complete rx/tx payloads                        \n\
                                                                                           \n\
  Client Args:                                                                             \n\
  -a                    -- Test with an async reply from server.                           \n\
                           Default is sync.                                                \n\
  -c [count]            -- Stop after sending [count] packets.                             \n\
                           Default is infinite.                                            \n\
  -e                    -- Server will reply on the error interface.                       \n\
                           Default use normal reply interface.                             \n\
  -f                    -- Flood  ping.  This will set -i to 0 and enable -r.              \n\
  -i [interval]         -- Wait  interval  seconds  between  sending  each                 \n\
                           packet.  The default is to wait for one second                  \n\
                           between each packet normally, or not to wait in                 \n\
                           flood mode. Default is 1 second.                                \n\
  -k                    -- Kill the client after a single reply.  This will                \n\
                           result in a new client for each request.  This makes            \n\
                           it easier to exercise the client creation code.                 \n\
                           Default is to use the same client for the whole test.           \n\
  -n                    -- Optional name to append to the default client name.             \n\
                           This is useful when running more than one client.               \n\
  -r                    -- This will ignore the reply from the server before               \n\
                           going to the next request/wait/etc.                             \n\
                           Default is to wait.                                             \n\
  -s [packetSize]       -- Specifies the number of data bytes to be sent.                  \n\
                           Default is 64. Minimum is 8 bytes.                              \n\
  -t                    -- Timestamp and track round trip times. Default is off.           \n\
  -w                    -- Wait on Exit. This is useful when using '-c' and                \n\
                           then client goes to exit.  This will prevent the                \n\
                           process from cleaning up to allow a chance to look              \n\
                           for potential leaks, etc. Default is no wait.                   \n\
                                                                                           \n\
  Server Args:                                                                             \n\
  -S                    -- Run as the server side. Default is client.                      \n\
                                                                                           \n\
  Example:                                                                                 \n\
  (NOTE: Adjust the verbosity to meet your needs)                                          \n\
  (NOTE: You can run multiple clients for a single server for more stress testing)         \n\
                                                                                           \n\
  # Stress/stability test of dbus.                                                         \n\
  # (Add '-s' for large packets sizes)                                                     \n\
  ping_stress2 -f -v                                                                       \n\
                                                                                           \n\
  # Look for memory leaks.                                                                 \n\
  # (Add '-a' and/or '-e' for other code paths in l2dbus/cdbus/etc)                        \n\
  ping_stress2 -c 50 -s 512000 -i 0 -w -v                                                  \n\
                                                                                           \n\
  # Performance testing.                                                                   \n\
  # (Add '-s' for other packets sizes, and adjust '-c' longer timing)                      \n\
  # NOTE: no verbosity so only calculate the timing and display the stats when done.       \n\
  # NOTE: Make sure the service instance has little or no verbosity.                       \n\
  ping_stress2 -c 50000 -t -i 0                                                            \n\
                                                                                           \n\
  # Performance testing, this will have the OS do the timing instead of us.                \n\
  # (Add '-s' for other packets sizes, and adjust '-c' longer timing)                      \n\
  # NOTE: Make sure the service instance has little or no verbosity.                       \n\
  time ping_stress2 -c 50000 -i 0                                                          \n\n\
"

/*---------------
  -- Structs
  ---------------*/

typedef struct DbusService
{
    cdbus_Connection *conn;
    cdbus_Dispatcher *disp;
    cdbus_Object     *obj;
    cdbus_Interface  *intf;
    cdbus_Bool       privConn;
    cdbus_Char       *svcName;
    cdbus_Handle     sigHnd;
} DbusService;

typedef struct DBusMsg
{
    struct cdbus_Connection *conn; /* used for async */
    struct DBusMessage      *msg;
    cdbus_Char              *payloadVal;
    cdbus_Char              *methodVal;

} DBusMsg;

typedef struct Stats
{
    unsigned long       iPkt;

    unsigned long       rx;
    unsigned long       rxErr;
    unsigned long       rxAsync;
    unsigned long       rxAsyncErr;

    unsigned long       tx;
    unsigned long       txErr;
    unsigned long       txAsync;
    unsigned long       txAsyncErr;

    unsigned long       avgRTT;
    unsigned long       minRTT;
    unsigned long       minRTTPkt;
    unsigned long       maxRTT;
    unsigned long       maxRTTPkt;
    unsigned long       totalRTT;  /* in nsec */
    unsigned long       totalTime; /* in msec */

    struct timespec     startTime; /* time test started */

} Stats;


/* Introspection data for public interfaces. */

static cdbus_DbusIntrospectArgs serviceMethodArgs[] =
{
    {"methodName",  "s", CDBUS_XFER_IN},
    {"args",        "s", CDBUS_XFER_IN},
    {"result",      "s", CDBUS_XFER_OUT}
};

static const cdbus_DbusIntrospectItem serviceMethods[] =
{
    { "Request",            serviceMethodArgs, 3},

    { "echo",               serviceMethodArgs, 3},
    { "echo_asError",       serviceMethodArgs, 3},
    { "echo_async",         serviceMethodArgs, 3},
    { "echo_asyncAsError",  serviceMethodArgs, 3}
};

enum { INVALID_FD = -1, READ_FD = 0, WRITE_FD, MAX_FD } eFD_NAMES;

enum {
    ECHO             = '.',
    ECHO_ERROR       = 'E',
    ECHO_ASYNC       = 'a',
    ECHO_ASYNC_ERROR = 'A'

} eCMD_NAMES;


/*---------------
  -- Globals
  ---------------*/

static struct ev_loop       *g_loop            = NULL;
static struct ev_io         g_pipeWatch;
static int                  g_cmdPipe[MAX_FD];

static cdbus_Dispatcher     *g_dispatcher      = NULL;
static cdbus_Connection     *g_conn            = NULL;
static DbusService          *g_dbusSvc         = NULL;
    
static pthread_t            g_clientThread     = 0;
static sem_t                g_exitSem;
static sem_t                g_clientSem;
static sem_t                g_replySem;

/*static char                 g_dbusBusName[128];*/

static Stats                g_stats;
static char                 *g_payload         = NULL;
static const char           *g_reqName         = "";

static cdbus_Bool           g_infiniteMode     = CDBUS_TRUE;

/* Command line options */
static cdbus_Bool           g_asyncMode        = CDBUS_FALSE; /* -a option                  */
static volatile int         g_iterCount        = 0;           /* -c option (0 is continuous)*/
static cdbus_Bool           g_errorIface       = CDBUS_FALSE; /* -e option                  */
static int                  g_waitInterval     = 1;           /* -i option                  */
static cdbus_Bool           g_killClients      = CDBUS_FALSE; /* -k option                  */
static char                 *g_clientName      = NULL;        /* -n option                  */                                         
static cdbus_Bool           g_ignoreReply      = CDBUS_FALSE; /* -r option                  */
static int                  g_payloadSize      = 64;          /* -s option                  */
static cdbus_Bool           g_serverMode       = CDBUS_FALSE; /* -S option                  */
static cdbus_Bool           g_timestamp        = CDBUS_FALSE; /* -t option                  */
static int                  g_verbose          = 0;           /* -v option                  */
static cdbus_Bool           g_waitOnExit       = CDBUS_FALSE; /* -w option                  */

    
/*---------------
  -- Macros
  ---------------*/
#define ARRAY_SIZE(X)   (sizeof(X) / sizeof(X[0]))
#define DIE(X)          do { fprintf(stderr, "%s(%d) ", __FILE__, __LINE__); die X; } while (0)

/*---------------
  -- Forward Decl
  ---------------*/
int  shutdownDbus();
void snapshot();
void destroyService( DbusService* );
DbusService* newService( cdbus_Bool );


/*!****************************************************************************
*
*  \b              die
*
*
*  Centralized routine to handle uniform logging of fatal errors then exit.
*
*  \param          [in]    format    ....
*  \param          [in]    ...       ....
*
*  \post           Does NOT return
*
******************************************************************************/
static void die( const char *format, ... )
{
    va_list     vargs;

    va_start(vargs, format);
    vfprintf(stderr, format, vargs);
    fprintf(stderr, "\n");
    exit( EXIT_FAILURE );

} /* die */

/*!****************************************************************************
*
*  \b              diffTime
*
*
*  Take the difference of two timestamps and return it in usec.
*
*  \param          [in]    startTime    ....
*  \param          [in]    endTime      ....
*
*  \return         time difference in usec
*
******************************************************************************/
static unsigned long diffTime( struct timespec *startTime, 
                               struct timespec *endTime )
{
    unsigned long    deltaSec, deltaUSec;
    
    if (endTime->tv_nsec > startTime->tv_nsec)
    {
        deltaSec  = (unsigned long)((endTime->tv_sec-startTime->tv_sec)*1000000);
    	deltaUSec = (unsigned long)((endTime->tv_nsec-startTime->tv_nsec)/1000);
    }
    else
    {
        deltaSec  = (unsigned long)((endTime->tv_sec-startTime->tv_sec-1)*1000000);
    	deltaUSec = (unsigned long)(1000000-(startTime->tv_nsec/1000)) + (unsigned long)(endTime->tv_nsec/1000);        
    }

    return( (unsigned long)(deltaSec + deltaUSec) );

} /* diffTime */

/*!****************************************************************************
*
*  \b              sigHandler
*
*
*  Signal handler to display current stats and attempt to cleanup DBus
*  before exiting the program.
*
*  \param          [in]    sig    ....
*
******************************************************************************/
static void sigHandler( int sig )
{
    signal(sig, SIG_IGN);

    if (g_timestamp)
    {
        struct timespec   endTime;
        clock_gettime(CLOCK_MONOTONIC, &endTime);
        g_stats.totalTime = diffTime( &g_stats.startTime, &endTime );
    }

    printf("\n");
    snapshot();
    printf("\n");

    if ( !g_serverMode )
    {
        /* Client */
        if ( g_clientThread > 0 )
        {
            pthread_cancel( g_clientThread );
            g_clientThread = 0;
        }
    }

    if (g_payload)
    {
        free( g_payload );
        g_payload = NULL;
    }

    cdbus_dispatcherStop(g_dispatcher);
    signal(sig, sigHandler);

} /* sigHandler */

/*!****************************************************************************
*
*  \b              snapshot
*
*
*  Sanpshot of the stats logged to stdout.
*
******************************************************************************/
void snapshot()
{
    printf("\nSTATS:\n");
    printf("========================================\n");
    printf("iPkt:       %ld\n", g_stats.iPkt);
    printf("rx:         %ld\n", g_stats.rx);
    printf("rxErr:      %ld\n", g_stats.rxErr);
    printf("rxAsync:    %ld\n", g_stats.rxAsync);
    printf("rxAsyncErr: %ld\n", g_stats.rxAsyncErr);
    printf("tx:         %ld\n", g_stats.tx);
    printf("txErr:      %ld\n", g_stats.txErr);
    printf("txAsync:    %ld\n", g_stats.txAsync);
    printf("txAsyncErr: %ld\n", g_stats.txAsyncErr);
    if (g_timestamp)
    {
        if ((!g_serverMode) && (!g_ignoreReply))
        {
            printf("minRTT:     %ld.%03ld (msec.usec) (pkt: %ld)\n", (g_stats.minRTT/1000), (g_stats.minRTT%1000), g_stats.minRTTPkt);
            printf("avgRTT:     %ld.%03ld (msec.usec)\n",            (g_stats.avgRTT/1000), (g_stats.avgRTT%1000));
            printf("maxRTT:     %ld:%06ld (sec:usec)  (pkt: %ld)\n", (g_stats.maxRTT/1000000), (g_stats.maxRTT%1000000), g_stats.maxRTTPkt);
            printf("totalRTT:   %ld:%06ld (sec:usec)\n",             (g_stats.totalRTT/1000000), (g_stats.totalRTT%1000000));
        }
        printf("Total Time: %ld:%06ld (sec:usec)\n", (g_stats.totalTime/1000000), (g_stats.totalTime%1000000));
    }
} /* snapshot */

/*!****************************************************************************
*
*  \b              traceMessage
*
*
*  Debug routine to dump the DBus message
*
*  \param          [in]    msg    ....
*
******************************************************************************/
static void traceMessage( struct DBusMessage* msg )
{
    const cdbus_Char    *msgTypeStr = "UNKNOWN";
    cdbus_Int32         msgType     = DBUS_MESSAGE_TYPE_INVALID;
    const cdbus_Char    *path       = NULL;
    const cdbus_Char    *intf       = NULL;
    const cdbus_Char    *name       = NULL;
    const cdbus_Char    *dest       = NULL;
    const cdbus_Char    *errName    = NULL;
    DBusMessageIter     dbusIter;
    cdbus_Int32         curDbusType;
    cdbus_Int32         argIdx      = 0;
    cdbus_Char          *argValue;

    if ( NULL != msg )
    {
        printf("%4ld. ", g_stats.iPkt);

        msgType    = dbus_message_get_type(msg);
        msgTypeStr = dbus_message_type_to_string(msgType);

        if ( (DBUS_MESSAGE_TYPE_METHOD_CALL == msgType) ||
             (DBUS_MESSAGE_TYPE_SIGNAL == msgType) )
        {
            path = dbus_message_get_path(msg);
            intf = dbus_message_get_interface(msg);
            name = dbus_message_get_member(msg);
            printf("(Ser=%u) [%s] <%s> %s%s%s\n",
                dbus_message_get_serial(msg),
                msgTypeStr,
                path ? path : "",
                intf ? intf : "",
                intf ? "." : "",
                name ? name : "");
        }
        else if (DBUS_MESSAGE_TYPE_METHOD_RETURN == msgType)
        {
            dest = dbus_message_get_destination(msg);
            printf("(RSer=%u) [%s] -> %s\n",
                        dbus_message_get_reply_serial(msg),
                        msgTypeStr,
                        dest ? dest : "");
        }
        else if (DBUS_MESSAGE_TYPE_ERROR == msgType )
        {
            errName = dbus_message_get_error_name(msg);
            printf("(RSer=%u) [%s] %s\n",
                    dbus_message_get_reply_serial(msg),
                    msgTypeStr,
                    errName ? errName : "");
        }
        else
        {
            printf("(Ser=%u) [%s]\n", dbus_message_get_serial(msg),
                  msgTypeStr);
        }

        if ( dbus_message_iter_init(msg, &dbusIter) )
        {
            while ( (curDbusType = dbus_message_iter_get_arg_type(&dbusIter)) != DBUS_TYPE_INVALID )
            {
                if ( (DBUS_TYPE_STRING == curDbusType) || (DBUS_TYPE_OBJECT_PATH == curDbusType) )
                {
                    dbus_message_iter_get_basic(&dbusIter, &argValue);
                    printf("      msg => Arg[%d]: %s\n", argIdx, argValue);
                }

                ++argIdx;
                dbus_message_iter_next(&dbusIter);
            }
        }
    }

} /* traceMessage */

/*!****************************************************************************
*
*  \b              handlePendingReply
*
*
*  Callback called for handling the reply.
*
*  \param          [in]    pending     ....
*  \param          [in]    userData    ....
*
******************************************************************************/
static void handlePendingReply( DBusPendingCall  *pending,
                                void             *userData )
{
    cdbus_Bool        bIsErr  = CDBUS_FALSE;
    cdbus_Int32       msgType = DBUS_MESSAGE_TYPE_INVALID;
    DBusMessage       *rspMsg = NULL;

    if ((g_timestamp) && (userData))
    {
    	struct timespec   endTime;
        unsigned long     execTime;
        struct timespec   *startTime = (struct timespec*)userData;
        clock_gettime(CLOCK_MONOTONIC, &endTime);
        
        execTime = diffTime( startTime, &endTime );

        g_stats.totalRTT += execTime;

        free( startTime );

        /* Max, Min, Avg */
        if (execTime > g_stats.maxRTT)
        {
            g_stats.maxRTT    = execTime;
            g_stats.maxRTTPkt = g_stats.iPkt;
        }
        
        if (execTime < g_stats.minRTT)
        {
            g_stats.minRTT    = execTime;
            g_stats.minRTTPkt = g_stats.iPkt;
        }
        
        if (g_stats.avgRTT == 0)
            g_stats.avgRTT = execTime;

        g_stats.avgRTT = (((g_stats.iPkt-1) * g_stats.avgRTT) + execTime)/g_stats.iPkt;
    }

    rspMsg = dbus_pending_call_steal_reply(pending);
    if (rspMsg)
    {
        msgType = dbus_message_get_type( rspMsg );
        if (DBUS_MESSAGE_TYPE_ERROR == msgType)
        {
            bIsErr = CDBUS_TRUE;
            g_stats.rxErr++;
        }
        else
        {
            g_stats.rx++;
        }
    }
    else
    {
        g_stats.rxErr++;
    }

    /* logging */
    if (g_verbose > 0)
    {
        if (g_verbose == 2)
        {
            if (bIsErr)
                printf("e");
            else
                printf("r");
        }
        else if ((g_verbose == 3) && (rspMsg))
        {
            traceMessage( rspMsg );        
        }
        fflush( stdout );
    }

    if (rspMsg)
        dbus_message_unref( rspMsg );

    sem_post( &g_clientSem ); /* Let the client continue */

} /* handlePendingReply */

/*!****************************************************************************
*
*  \b              clientRequests
*
*
*  Thread handler for all client requests.
*
*  \param          [in]    usr_iface_ptr    ....user data for thread
*
******************************************************************************/
static void *clientRequests( void *usr_iface_ptr )
{
    char        cmd;

    g_payload = malloc(g_payloadSize);
    assert( NULL != g_payload );

    if ((g_asyncMode == CDBUS_TRUE) && (g_errorIface == CDBUS_TRUE))
    {
        g_reqName = "echo_asyncAsError";
        cmd       =  ECHO_ASYNC_ERROR;        
    }
    else if (g_asyncMode == CDBUS_TRUE)
    {
        g_reqName = "echo_async";
        cmd       =  ECHO_ASYNC;
    }
    else if (g_errorIface == CDBUS_TRUE)
    {
        g_reqName = "echo_asError";
        cmd       =  ECHO_ERROR;
    }
    else
    {
        g_reqName = "echo";
        cmd       =  ECHO;
    }
    memset(g_payload, cmd, g_payloadSize);

    if (g_waitOnExit == CDBUS_TRUE)
    {
        printf("----------------------------------------------------\n");
        printf("PRESS <ENTER> TO START...  (pid: %d)\n",  (int) getpid());
        printf("----------------------------------------------------\n");
        getchar();
    }

    if (g_verbose == 1)
    {
        printf("Pkt: 0");
        fflush( stdout );
    }

    if (!g_infiniteMode)
    {
        /* Finite count mode, this is used to ensure we are  */
        /* done sending before this thread exits.            */
        sem_init( &g_exitSem, 0, 0);
    }
    sem_init( &g_clientSem, 0, 0);
    sem_init( &g_replySem, 0, 0);

    if (g_timestamp)
        clock_gettime(CLOCK_MONOTONIC, &g_stats.startTime);

    while ((g_infiniteMode == CDBUS_TRUE) || (g_iterCount > 0))
    {
        g_stats.iPkt++;

        if (g_killClients == CDBUS_TRUE && g_stats.iPkt != 1)
        {
            destroyService( g_dbusSvc );
            newService(CDBUS_FALSE);
        }

        /*----------------------------
          -- Pre Stats
          ----------------------------*/
        if ((g_asyncMode == CDBUS_TRUE) && (g_errorIface == CDBUS_TRUE))
            g_stats.txAsyncErr++;
        else if (g_asyncMode == CDBUS_TRUE)
            g_stats.txAsync++;
        else if (g_errorIface == CDBUS_TRUE)
            g_stats.txErr++;
        else
            g_stats.tx++;

        /*----------------------------------------------------*/
        /*-- Do request                                       */
        if (write(g_cmdPipe[WRITE_FD], (void*)(&cmd), sizeof(cmd)) < 1)
        {
            printf("ERROR: Failed to signal the request to execute.\n");
        }
        /*----------------------------------------------------*/
        sem_wait( &g_clientSem ); /* Wait for the cmd to be sent */

        if (g_waitInterval > 0)
        {
            if (g_verbose == 2)
            {
                printf(".");
                fflush( stdout );
            }

            sleep( g_waitInterval );
        }

        if (g_iterCount > 0)
            g_iterCount--;

    } /* end loop */

    if (!g_infiniteMode)
    {
        /* Wait for the sending to complete before we exit */
        sem_wait( &g_exitSem );
    }

    if (g_timestamp)
    {
        struct timespec   endTime;
        clock_gettime(CLOCK_MONOTONIC, &endTime);
        g_stats.totalTime = diffTime( &g_stats.startTime, &endTime );
    }

    printf("\n");
    snapshot();  /* display stats */
    printf("\n");

    shutdownDbus();

    if (g_waitOnExit == CDBUS_TRUE)
    {
        printf("----------------------------------------------------\n");
        printf("PRESS <ENTER> TO EXIT...  (pid: %d)\n", (int) getpid());
        printf("----------------------------------------------------\n");
        getchar();
    }

    if (g_payload)
    {
        free( g_payload );
        g_payload = NULL;
    }

    return( NULL );

} /* clientRequests */

/*!****************************************************************************
*
*  \b              cmdPipe_cb
*
*
*  Callback to handler the reads on the command pipe.
*
*  \param          [in]    watch      ....N/A
*  \param          [in]    revents    ....mask of event that triggered
*
******************************************************************************/
static void cmdPipe_cb(EV_P_ struct ev_io *watch, int revents)
{
    char              cmd;
    DBusPendingCall   *pending    = NULL;
    DBusMessage       *reqMsg     = NULL;
    cdbus_Bool        ret;
    struct timespec   *pStartTime = NULL;

    (void)watch;

    if ((revents & EV_READ) == 0)
    {
        return;
    }

    while ( sizeof(cmd) == read(g_cmdPipe[READ_FD], (void*)(&cmd), sizeof(cmd)) )
    {
        reqMsg = dbus_message_new_method_call(  STRESS_BUSNAME,
                                                STRESS_OBJPATH,
                                                STRESS_INTERFACE,
                                                "Request" );
        assert( NULL != reqMsg );
        dbus_message_append_args( reqMsg,
                                  DBUS_TYPE_STRING, &g_reqName,
                                  DBUS_TYPE_STRING, &g_payload,
                                  DBUS_TYPE_INVALID );
        /* logging */
        if (g_verbose > 0)
        {
            if (g_verbose == 1)
                printf("\rPkt: %ld", g_stats.iPkt);
            else if (g_verbose == 2)
                printf("t");
            else if (g_verbose == 3)
                traceMessage(reqMsg);

            fflush( stdout );
        }

        if (g_ignoreReply)
        {
            ret = dbus_connection_send(cdbus_connectionGetDBus(g_conn), reqMsg, NULL);
            if (ret)
                dbus_connection_flush(cdbus_connectionGetDBus(g_conn));
        }
        else
        {
            if (g_timestamp)
            {
                pStartTime = malloc( sizeof(struct timespec) );
                clock_gettime(CLOCK_MONOTONIC, pStartTime);
            }

            ret = cdbus_connectionSendWithReply( g_conn, reqMsg, &pending, 
                                                 ASYNC_REPLY_TIMEOUT, 
                                                 handlePendingReply, (void*)pStartTime, NULL);
        }
        dbus_message_unref(reqMsg);

        if ( !ret )
        {
            g_stats.rxErr++;
            
            if (!g_ignoreReply)
            {
                dbus_pending_call_unref( pending );
                pending = NULL;                
            }
        }
        else if (!g_ignoreReply)
        {
            dbus_pending_call_unref( pending );
            pending = NULL;
        }
        
        if (g_ignoreReply)
        {
            sem_post( &g_clientSem ); /* Let the client continue */
        }
        /* else, the reply handler will unblock this */

    } /* end while */

    if (!g_infiniteMode)
    {
        /* Finite count, Notify the client thread we are done */
        sem_post( &g_exitSem );
    }

} /* cmdPipe_cb */

/*!****************************************************************************
*
*  \b              wakeup
*
*
*  [add description here]
*
*  \param          [in]    disp        ....
*  \param          [in]    userData    ....
*
******************************************************************************/
static void wakeup ( cdbus_Dispatcher   *disp,
                     void               *userData )
{
    (void)userData;
    cdbus_dispatcherInvokePending(disp);
} /* wakeup */

/*!****************************************************************************
*
*  \b              destroyService
*
*
*  Destroy all service specific references
*
*  \param          [in]    svc    ....
*
******************************************************************************/
void destroyService( DbusService* svc )
{
    cdbus_HResult result;

    if ( NULL != svc )
    {
        if ( CDBUS_INVALID_HANDLE != svc->sigHnd )
        {
            result = cdbus_connectionUnregMatchHandler(svc->conn, svc->sigHnd);
            if ( CDBUS_FAILED(result) )
            {
                fprintf(stderr, "ERROR: Failed to unregister signal: 0x%X\n", result);
            }
        }

        if ( !cdbus_connectionUnregisterObject(svc->conn, STRESS_OBJPATH) )
        {
            fprintf(stderr,"ERROR: Failed to unregister the service object!\n");
        }

        if ( svc->privConn )
        {
            cdbus_connectionClose(svc->conn);
        }

        cdbus_interfaceUnref(svc->intf);
        cdbus_objectUnref(svc->obj);
        cdbus_connectionUnref(svc->conn);
        cdbus_dispatcherUnref(svc->disp);
        free(svc->svcName);
        free(svc);
    }
} /* destroyService */

/*!****************************************************************************
*
*  \b              echo
*
*
*  Echo the original payload using the normal reply interface.
*
*  \param          [in]    tMsg      ....
*  \param          [in]    bAsync    ....true, being called from the Async handler
*                                        false, being called from main handler
*
*  \return         [add here]
*
******************************************************************************/
static DBusMessage* echo( struct DBusMsg *tMsg, cdbus_Bool bAsync )
{
    DBusMessage     *replyMsg = NULL;

    if (!bAsync)
    {
        g_stats.iPkt++;

        /* do both since we do both */
        g_stats.rx++;
        g_stats.tx++;
    }

    replyMsg = dbus_message_new_method_return( tMsg->msg );
    assert( NULL != replyMsg );
    dbus_message_append_args( replyMsg,
                              DBUS_TYPE_STRING, &tMsg->payloadVal,
                              DBUS_TYPE_INVALID );
    /* logging */
    if (g_verbose > 0)
    {
        if (g_verbose == 1)
            printf("\rPkt: %ld", g_stats.iPkt);
        else if (g_verbose == 2)
        {
            if (bAsync)
                printf("t");
            else
                printf(".");
        }
        else if ((g_verbose == 3) && (replyMsg))
        {
            traceMessage( replyMsg );        
        }
        fflush( stdout );
    }
    
    dbus_message_unref(tMsg->msg);

    return( replyMsg );

} /* echo */

/*!****************************************************************************
*
*  \b              echo_asError
*
*
*  Echo the original payload using the error interface.
*
*  \param          [in]    tMsg      ....
*  \param          [in]    bAsync    ....true, being called from the Async handler
*                                        false, being called from main handler
*
*  \return         [add here]
*
******************************************************************************/
static DBusMessage* echo_asError( struct DBusMsg* tMsg, cdbus_Bool bAsync )
{
    DBusMessage     *replyMsg = NULL;

    if (!bAsync)
    {
        g_stats.iPkt++;

        /* do both since we do both */
        g_stats.rxAsync++;
        g_stats.txAsync++;
    }

    replyMsg = dbus_message_new_error_printf( tMsg->msg,
                                              STRESS_ERRORIFACE,
                                              tMsg->payloadVal );
    assert( NULL != replyMsg );

    /* logging */
    if (g_verbose > 0)
    {
        if (g_verbose == 1)
            printf("\rPkt: %ld", g_stats.iPkt);
        else if (g_verbose == 2)
        {
            if (bAsync)
                printf("T");
            else
                printf("e");
        }
        else if ((g_verbose == 3) && (replyMsg))
        {
            traceMessage( replyMsg );        
        }
        fflush( stdout );
    }

    dbus_message_unref(tMsg->msg);

    return( replyMsg );

} /* echo_asError */

/*!****************************************************************************
*
*  \b              echo_asyncHandler
*
*
*  Timer handler to ultimately send the async response.
*
*  \param          [in]    hTimer    ....
*  \param          [in]    user      ....
*
*  \return         [add here]
*
******************************************************************************/
static cdbus_Bool echo_asyncHandler( cdbus_Timeout  *hTimer,
                                     void           *user )
{
    DBusMessage     *replyMsg = NULL;
    struct DBusMsg  *tMsg     = (struct DBusMsg*)user;

    if ( !strcmp(tMsg->methodVal, "echo_async") )
    {
        replyMsg = echo( tMsg, CDBUS_TRUE );
        g_stats.txAsync++;
    }
    else if ( !strcmp(tMsg->methodVal, "echo_asyncAsError") )
    {
        replyMsg = echo_asError( tMsg, CDBUS_TRUE );        
        g_stats.txAsyncErr++;
    }
    else
    {
        dbus_message_unref( tMsg->msg );
    }

    if ( replyMsg )
    {
        dbus_connection_send( cdbus_connectionGetDBus(tMsg->conn),
                              replyMsg, NULL);
        dbus_message_unref(replyMsg);
    }

    cdbus_connectionUnref( tMsg->conn );

    free(tMsg);

    return( CDBUS_FALSE );

} /* echo_asyncHandler */

/*!****************************************************************************
*
*  \b              echo_asyncGeneric
*
*
*  Handler for all async echo calls.  This will setup a short timer that
*  will then send the response (i.e. reply asynchronously)
*
*  \param          [in]    tMsg    ....
*
*  \return         None
*
******************************************************************************/
static void echo_asyncGeneric( struct DBusMsg *tMsg )
{
    cdbus_Bool      bAsErr  = CDBUS_FALSE;
    cdbus_HResult   rc;
    struct DBusMsg  *tMsgCopy;
    cdbus_Timeout   *hTimer = NULL;

    g_stats.iPkt++;

    if ( !strcmp(tMsg->methodVal, "echo_async") )
    {
        g_stats.rxAsync++;
    }
    else if ( !strcmp(tMsg->methodVal, "echo_asyncAsError") )
    {
        bAsErr = CDBUS_TRUE;
        g_stats.rxAsyncErr++;
    }

    /* logging */
    if (g_verbose > 0)
    {
        if (g_verbose == 1)
            printf("\rPkt: %ld", g_stats.iPkt);
        else if (g_verbose == 2)
        {
            if (bAsErr)
                printf("R");
            else
                printf("r");
        }
        else if (g_verbose == 3)
            traceMessage( tMsg->msg );        
        fflush( stdout );
    }

    tMsgCopy = malloc( sizeof(struct DBusMsg) );
    memcpy((unsigned char*)tMsgCopy, (unsigned char*)tMsg, sizeof(struct DBusMsg));

    hTimer = cdbus_timeoutNew( g_dispatcher, 
                               ASYNC_TIMEOUT, 
                               CDBUS_FALSE,
                               echo_asyncHandler, (void*)tMsgCopy);
    if ( NULL == hTimer )
    {
        DIE(("ERROR: Failed to create timeout watch!\n"));
    }

    rc = cdbus_timeoutEnable( hTimer, CDBUS_TRUE );
    if ( CDBUS_FAILED(rc) )
    {
        DIE(("ERROR: Failed to enable timeout watch!\n"));
    }

} /* echo_asyncGeneric */

/*!****************************************************************************
*
*  \b              objectMessageHandler
*
*
*  Main handler for all DBus message processing.
*
*  \param          [in]    obj     ....
*  \param          [in]    conn    ....
*  \param          [in]    msg     ....
*
*  \return         [add here]
*
******************************************************************************/
static DBusHandlerResult objectMessageHandler( struct cdbus_Object      *obj,
                                               struct cdbus_Connection  *conn,
                                               DBusMessage              *msg )
{
    DBusHandlerResult   result         = DBUS_HANDLER_RESULT_NOT_YET_HANDLED;
    const cdbus_Char    *member        = NULL;
    DBusMsg             dbusMsg;
    DBusMessage         *replyMsg      = NULL;
    int                 msgType        = dbus_message_get_type(msg);

    /* logging */
    if (g_verbose > 0)
    {
        if (g_verbose == 2)
            printf("r");
        else if ((g_verbose == 3) && (msg))
            traceMessage( msg );        
        fflush( stdout );
    }

    if (DBUS_MESSAGE_TYPE_METHOD_CALL == msgType )
    {
        member = dbus_message_get_member(msg);

        /* Later implement DBus interfaces for testing */
        #if 0
        if ( !strcmp(member, "echo") )
            echo( msg, NULL, NULL );
        else if ( !strcmp(member, "echo_asError") )
            echo_asError( msg, NULL, NULL );
        else if ( !strcmp(member, "echo_async") )
            echo_async( msg, NULL, NULL );
        else if ( !strcmp(member, "echo_asyncAsError") )
            echo_asyncAsError( msg, NULL, NULL );
        else
        #endif

            if ( !strcmp(member, "Request") )
        {
            /* Expects: arg[0] == string == method Name,
                        arg[1] == string == payload */

            cdbus_Int32     curDbusType;
            DBusMessageIter dbusIter;
            cdbus_Char      *methodVal  = NULL;
            cdbus_Char      *payloadVal = NULL;

            /* Parse the first two arguments */
            if ( dbus_message_iter_init(msg, &dbusIter) )
            {
                curDbusType = dbus_message_iter_get_arg_type(&dbusIter);
                if ( (DBUS_TYPE_STRING == curDbusType) || (DBUS_TYPE_OBJECT_PATH == curDbusType) )
                {
                    dbus_message_iter_get_basic(&dbusIter, &methodVal);
                    
                    dbus_message_iter_next(&dbusIter);
                    curDbusType = dbus_message_iter_get_arg_type(&dbusIter);
                    if ( (DBUS_TYPE_STRING == curDbusType) || (DBUS_TYPE_OBJECT_PATH == curDbusType) )
                    {
                        dbus_message_iter_get_basic(&dbusIter, &payloadVal);
                    }
                }
            }

            if ((methodVal == NULL) || (payloadVal == NULL))
            {
                replyMsg = dbus_message_new_error_printf( msg,
                                        STRESS_ERRORIFACE,
                                        "Invalid args for Request()");
                assert( NULL != replyMsg );
            }
            else
            {
                dbus_message_ref( msg );
                
                dbusMsg.msg         = msg;
                dbusMsg.methodVal   = methodVal;
                dbusMsg.payloadVal  = payloadVal;

                if ( !strcmp(methodVal, "echo") )
                    replyMsg = echo( &dbusMsg, CDBUS_FALSE );
                else if ( !strcmp(methodVal, "echo_asError") )
                    replyMsg = echo_asError( &dbusMsg, CDBUS_FALSE );
                else if (( !strcmp(methodVal, "echo_async") ) ||
                         ( !strcmp(methodVal, "echo_asyncAsError") ))
                {
                    cdbus_connectionRef( conn );
                    dbusMsg.conn = conn;
                    echo_asyncGeneric( &dbusMsg );
                }
                else
                {
                    dbus_message_unref( msg );
                }
            }
        }
        else
        {
            replyMsg = dbus_message_new_error_printf(msg, STRESS_ERRORIFACE,
                                                     "Unknown method (%s)", member);
            assert( NULL != replyMsg );
        }

        if (replyMsg)
        {
            dbus_connection_send( cdbus_connectionGetDBus(conn), replyMsg, NULL);
            dbus_message_unref(replyMsg);
        }

        result = DBUS_HANDLER_RESULT_HANDLED;
    }
    else if ( DBUS_MESSAGE_TYPE_SIGNAL == msgType )
    {
        member = dbus_message_get_member(msg);
        printf("Received signal: %s\n", member);
        result = DBUS_HANDLER_RESULT_HANDLED;
    }

    return( result );

} /* objectMessageHandler */

/*!****************************************************************************
*
*  \b              serviceProviderInterfaceHandler
*
*
*  [add description here]
*
*  \param          [in]    conn        ....
*  \param          [in]    obj         ....
*  \param          [in]    msg         ....
*  \param          [in]    userData    ....
*
*  \return         [add here]
*
******************************************************************************/
static DBusHandlerResult serviceProviderInterfaceHandler(
                                        struct cdbus_Connection *conn,
                                        struct cdbus_Object     *obj,
                                        DBusMessage             *msg,
                                        void                    *userData )
{
    return objectMessageHandler(obj, conn, msg);

} /* serviceProviderInterfaceHandler */

/*!****************************************************************************
*
*  \b              newService
*
*
*  Initialize the DBus service.
*
*  \param          [in]    privConn    ....
*
*  \return         struct with all service information needed.
*
******************************************************************************/
DbusService* newService( cdbus_Bool  privConn )
{
    cdbus_Interface     *introspectIntf;
    DbusService         *svc = calloc(1, sizeof(*svc));

    assert( NULL != svc );

    svc->privConn = privConn;
    svc->svcName  = strdup("PingStress");

    svc->disp = cdbus_dispatcherNew(EV_DEFAULT_ CDBUS_TRUE, wakeup, NULL);
    assert( NULL != svc->disp );

    introspectIntf = cdbus_introspectNew();

    if ( NULL == introspectIntf )
    {
        DIE(("Failed to create introspection interface!"));
    }
    
    svc->obj = cdbus_objectNew( STRESS_OBJPATH,
                                objectMessageHandler,
                                svc->svcName);
    if ( NULL == svc->obj )
    {
        DIE(("Failed to create object!"));
    }

    if ( !cdbus_objectAddInterface(svc->obj, introspectIntf) )
    {
        DIE(("ERROR: Failed to add an interface to object!"));
    }

    /* We no longer need a reference to the introspection interface */
    cdbus_interfaceUnref( introspectIntf );

    svc->intf = cdbus_interfaceNew( STRESS_INTERFACE,
                                    serviceProviderInterfaceHandler,
                                    "ServiceProvider");
    if ( NULL == svc->intf )
    {
        DIE(("ERROR: Failed to create new interface!"));
    }

    if ( !cdbus_interfaceRegisterMethods(svc->intf, serviceMethods,
                                         ARRAY_SIZE(serviceMethods)) )
    {
        DIE(("ERROR: Failed to register interface methods!"));
    }

    if ( !cdbus_objectAddInterface(svc->obj, svc->intf) )
    {
        DIE(("ERROR: Failed to add an interface to object!"));
    }

    /* Open a connection */
    svc->conn = cdbus_connectionOpenStandard(svc->disp, 
                                             DBUS_BUS_SESSION,
                                             privConn, CDBUS_FALSE);
    if ( NULL == svc->conn )
    {
        DIE(("ERROR: Failed to open a connection."));
    }

    if ( !cdbus_connectionRegisterObject(svc->conn, svc->obj) )
    {
        DIE(("ERROR: Failed to register object with connection!"));
    }
    return( svc );

} /* newService */

/*!****************************************************************************
*
*  \b              initCmdPipe
*
*
*  Initialize the command pipe used to send DBus commands from the client
*  thread to the DBus thread.  Since CDBus is not thread safe we need to
*  ensure the DBus commands occur on the DBus thread.
*
*  \param          None
*
*  \pre            None
*
*  \post           None
*
*  \return         None
*
******************************************************************************/
void initCmdPipe()
{
    int ret;

    g_cmdPipe[READ_FD]  = INVALID_FD;
    g_cmdPipe[WRITE_FD] = INVALID_FD;

    ret = pipe( g_cmdPipe );
    if ( 0 != ret )
    {
        DIE(("ERROR: Failed to create the command pipe!"));
    }

    fcntl(g_cmdPipe[READ_FD],  F_SETFL, O_NONBLOCK);
    fcntl(g_cmdPipe[WRITE_FD], F_SETFL, O_NONBLOCK);

    ev_io_init( &g_pipeWatch, cmdPipe_cb, g_cmdPipe[READ_FD], EV_READ );
    ev_io_start( g_loop, &g_pipeWatch );

    ret = pthread_create( &g_clientThread, NULL, 
                          clientRequests,  NULL);
    if (ret != 0)
    {
        DIE(("ERROR: Unable to create client thread\n"));
    }

} /* initCmdPipe */

/*!****************************************************************************
*
*  \b              initDbus
*
*
*  Do all DBus initialization
*
******************************************************************************/
void initDbus()
{
    cdbus_Bool          privConn = CDBUS_FALSE;
    cdbus_HResult       rc;

    rc = cdbus_initialize();

    if ( CDBUS_FAILED(rc) )
    {
        DIE(("ERROR: Failed to initialize library (rc=0x%X).", rc));
    }

    /* New dispatcher */
    if (!g_serverMode)
    {
        g_loop       = ev_default_loop (0);
        g_dispatcher = cdbus_dispatcherNew(g_loop, CDBUS_FALSE, wakeup, NULL);
    }
    else
    {
        /* Uses the default loop internally */
        g_dispatcher = cdbus_dispatcherNew(EV_DEFAULT_ CDBUS_FALSE, wakeup, NULL);
    }

    if ( NULL == g_dispatcher )
    {
        DIE(("ERROR: Failed to create dispatcher.\n"));
    }

    /* Open a connection */
    g_conn = cdbus_connectionOpenStandard( g_dispatcher, 
                                           DBUS_BUS_SESSION,
                                           privConn, 
                                           CDBUS_FALSE);

    if ( NULL == g_conn )
    {
        DIE(("ERROR: Failed to open a connection.\n"));
    }

    if ( g_serverMode )
    {
        /* SERVER MODE */
        /*-------------*/

        /* Register the bus name */
        if ( DBUS_REQUEST_NAME_REPLY_PRIMARY_OWNER !=
            dbus_bus_request_name( cdbus_connectionGetDBus(g_conn), 
                                   STRESS_BUSNAME,
                                   DBUS_NAME_FLAG_DO_NOT_QUEUE, NULL) )
        {
            DIE(("ERROR: Failed to register bus name (%s)\n", STRESS_BUSNAME));
        }
        
        g_dbusSvc = newService( privConn );
        if ( NULL == g_dbusSvc )
        {
            DIE(("ERROR: Failed to create service\n"));
        }

        if (g_timestamp)
            clock_gettime(CLOCK_MONOTONIC, &g_stats.startTime);
    }
    else
    {
        /* CLIENT MODE */
        /*-------------*/

        char clientName[256];

        if (g_clientName)
            snprintf(clientName, 256, "%sClient%s", STRESS_BUSNAME, g_clientName);
        else
            snprintf(clientName, 256, "%sClient", STRESS_BUSNAME);

        /* Register the bus name */
        if ( DBUS_REQUEST_NAME_REPLY_PRIMARY_OWNER !=
            dbus_bus_request_name( cdbus_connectionGetDBus(g_conn), 
                                   clientName,
                                   DBUS_NAME_FLAG_DO_NOT_QUEUE, NULL) )
        {
            DIE(("ERROR: Failed to register bus name (%s)\n", clientName));
        }

        if (g_clientName)
            free(g_clientName);

        /* Command Pipe Approach */
        /* All DBus communication should be on the main thread */
        initCmdPipe();
    }
} /* initDbus */

/*!****************************************************************************
*
*  \b              parseArgs
*
*
*  Parse command line arguments.
*
*  \param          [in]    argc    ....arg count
*  \param          [out]   argv    ....arg array
*
******************************************************************************/
static void parseArgs(int argc, char **argv)
{
    int 		opt;
    cdbus_Bool  bHelp = CDBUS_FALSE;

    while (1)
    {
        int option_index = 0;

        static struct option long_options[] = {
            { "bus",     required_argument, NULL,  'c' },
            { 0,         0,                 0,     0 }
        };

        opt = getopt_long(argc, argv, "ab:c:efghi:kn:rs:Stvw",
                          long_options, &option_index);
        if (opt == -1) 
            break;

        switch (opt)
        {
        case 'a':
            g_asyncMode = CDBUS_TRUE;
            break;

        #if 0
        case 'b':
            snprintf(g_dbusBusName, 128, "%s", optarg);
            break;
        #endif

        case 'c':
            g_iterCount = atol(optarg);
            if (g_iterCount < 1)
            {
                DIE(("ERROR: Invalid iteration count (option -c): %s\n", optarg));
            }
            g_infiniteMode = CDBUS_FALSE;
            break;

        case 'e':
            g_errorIface = CDBUS_TRUE;
            break;

        case 'f':
            g_waitInterval = 0;
            g_ignoreReply  = CDBUS_TRUE;
            break;

        case 'h':
            bHelp = CDBUS_TRUE;
            break;

        case 'i':
            g_waitInterval = atol(optarg);
            if (g_waitInterval < 0)
            {
                DIE(("ERROR: Invalid wait interval (option -i): %s\n", optarg));
            }
            break;

        case 'k':
            g_killClients = CDBUS_TRUE;
            break;

        case 'n':
            g_clientName = malloc( strlen(optarg)+1 );
            strcpy(g_clientName, optarg);
            break;

        case 'r':
            g_ignoreReply = CDBUS_TRUE;
            break;

        case 's':
            g_payloadSize = atol(optarg);
            if (g_payloadSize < 1)
            {
                DIE(("ERROR: Invalid payload size (option -s): %s\n", optarg));
            }
            break;

        case 'S':
            g_serverMode = CDBUS_TRUE;
            break;

        case 't':
            g_timestamp = CDBUS_TRUE;
            break;

        case 'v':
            g_verbose++;
            break;

        case 'w':
            g_waitOnExit = CDBUS_TRUE;
            break;

        }
    } /* end arg loop */

    if (g_verbose > 0) 
    {
        printf("\n");
        printf("Version:                            %s\n", APP_VER);

        #if 0
        if (g_dbusBusName != dbusUtils.SESSION_BUS)
            printf("Bus Name:                           %s\n", g_dbusBusName);
        #endif

        printf("Running as Server:                  %s\n", (g_serverMode)?"TRUE":"false");
        printf("Verbose:                            %d\n", g_verbose); 
        printf("Timestamp Performance:              %s\n", (g_timestamp)?"TRUE":"false");

        if (g_serverMode)
        {
            printf("\n");
            printf("NOTE: Connect to Bus: %s\n", STRESS_BUSNAME);
        }
        else /* client */
        {
            printf("Client Name:                        %sClient%s\n", STRESS_BUSNAME, (g_clientName)?g_clientName:"");
            printf("Test Async Server Reply:            %s\n", (g_asyncMode  )?"TRUE":"false");
            printf("Ingnore Reply from Request:         %s\n", (g_ignoreReply)?"TRUE":"false");
            printf("Reply Using Error Interface:        %s\n", (g_errorIface )?"TRUE":"false");
            printf("Create/Kill Client After Replies:   %s\n", (g_killClients)?"TRUE":"false");
            printf("Payload Size:                       %d\n", g_payloadSize);

            if (g_iterCount == 0)
                printf("Iteration Count:                    infinite\n");
            else
                printf("Iteration Count:                    %d\n", g_iterCount);

            printf("Interpacket Wait Interval:          %d (sec)\n", g_waitInterval);

            if ((g_ignoreReply == CDBUS_TRUE) && (g_waitInterval == 0))
                printf("FLOOD MODE:                        TRUE\n");
        }
        printf("\n");
    }

    if (bHelp)
    {
        /* this way you can see what is parsed prior to help */
        printf("\n");
        printf("%s %s\n", argv[0], APP_VER);
        printf(HELP_TEXT);
        if (g_clientName)
            free(g_clientName);
        exit(1);
    }

} /* parseArgs */

/*!****************************************************************************
*
*  \b              shutdownDbus
*
*
*  Attempts to cleanly shutdown D-Bus and deallocate resources.
*  The main event loop should exit as well.
*
******************************************************************************/
int shutdownDbus()
{
    /* Destroy the service objects */
    destroyService( g_dbusSvc );

    cdbus_connectionClose( g_conn );
    cdbus_connectionUnref( g_conn );

    /* FIXME: Need some thought/work for clean shutdown */
    cdbus_dispatcherUnref( g_dispatcher );
    g_dispatcher = NULL;

    return( CDBUS_SUCCEEDED( cdbus_shutdown() ) ? EXIT_SUCCESS : EXIT_FAILURE );

} /* shutdownDbus */

/*!****************************************************************************
*
*  \b              main
* 
******************************************************************************/
int main(int argc, char **argv)
{
    parseArgs( argc, argv );

    memset( (void*)&g_stats, 0, sizeof(Stats) );
    g_stats.minRTT = 0xFFFFFFFF;

    signal(SIGINT,  sigHandler);
    signal(SIGTERM, sigHandler);

    initDbus();

    /*-----------------------------------------------------*/
    /* Run the main loop                                   */
    /*-----------------------------------------------------*/
    cdbus_dispatcherRun(g_dispatcher, CDBUS_RUN_WAIT);

    shutdownDbus();

    return( EXIT_SUCCESS );

} /* main */
