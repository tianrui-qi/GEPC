# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 14:20:34 2021

@author: jeanbaptiste
"""

import time
from threading import Thread
from queue import Queue
import gc
import socket
import pickle
import sys
import os
import traceback

import numpy as np
import tensorflow as tf

class Server(Thread):
    """A feedback control server daemon.
    
    This server monitors its inputs queue, and when there are some it compiles
    them together and sends them to the feedback controller
    """
    
    def __init__(self, controller, device = None):
        """
        Initialize object

        Parameters
        ----------
        controller : deepcellcontrol.core.control.control.Controller object
            A controller object that will be performing control computation
            for all clients. Must have a .feedback() method
        device : str or None, optional
            Tensorflow device. Can be '/device:CPU:0', '/device:GPU:0' or
            logical GPUs. If None, no specific device is assigned. 
            See tf.device()

        Returns
        -------
        None.

        """
        super().__init__(daemon=True)
        
        self.controller = controller
        "Feedback controller. From the control module. Must have a feedback method"
        self.queue = Queue()
        """Inputs FIFO queue, use .put() to add to it. 
        See `queue.Queue <https://docs.python.org/3/library/queue.html>`_
        """
        self.device = device
        """Tensorflow device. Can be '/device:CPU:0', '/device:GPU:0' or
        logical GPUs. If None, no specific device is assigned. See tf.device()
        """
        self.batch_size = 3
        """Number of positions inputs to process at once. This is a trade-off, 
        we process larger batches faster but we have to wait for all positions
        to be done when we could be doing other operations"""
        self._stop_flag = False
        "Flag to trigger the daemon to stop, for debugging mostly"
        self.verbose = True
        "Whether to print messages to console"
        self.name = ""
        "Server name for printing"
    
    def run(self):
        """
        Daemon loop, constantly running once the thread is started. See
        `threading.Thread <https://docs.python.org/3/library/threading.html>`_
        
        When inputs are available, they are retrieved from the queue, sent
        to run_feedback(), and then sent back to the "client" through its
        dispatcher method, along with any metadata the client sent with it.
        
        The metadata are not useful for the server, but the "client" might need
        the associated metadata when it handles the outputs after they have
        been dispatched back. They can be anything, an int, a list, a dict...

        Returns
        -------
        None.

        """
        
        while not self._stop_flag:
            
            time.sleep(.01)
            
            self._run()
        
        # Reset flag
        self._stop_flag = False
    
    def _msg(self, msg):
        
        if self.verbose:
            print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} - server {self.name} - {msg}")
            
    
    def _run(self):
        """
        Function that performs the actual run operations.
        
        Done for debugging purposes.

        Returns
        -------
        None.

        """
        
        # If queue is empty, nothing to do
        if self.queue.empty():
            return
        
        # Retrieve inputs from queue:
        feedback_inputs, metadata, output_dispatchers = self.get_inputs()
            
        # Send out to controller
        control_outputs = self.run_feedback(feedback_inputs)
        
        # Dispatch:
        self.dispatch(control_outputs, metadata, output_dispatchers)
        
        del feedback_inputs
        del control_outputs
        del metadata
        del output_dispatchers
        gc.collect()
    
    def get_inputs(self):
        """
        Retrieve inputs from queue

        Returns
        -------
        feedback_inputs : List
            List of inputs and objectives to pass to the controller's
            feedback() method.
        metadata : List of objects
            List of metadata objects. These are not used by the server 
            but simply passed back to "client" via the dispatcher for reference.
            They can be anything
        output_dispatchers : List
            List of callable. These will be called to dispatch control output
            results.

        """
        
        # Empty inputs lists:
        feedback_inputs = []
        metadata = []
        output_dispatchers = []
        
        # Get items (if any) and append to lists:
        while not self.queue.empty() and len(feedback_inputs) < self.batch_size:
            inputs, meta, dispatcher = self.queue.get()
            feedback_inputs += [inputs]
            metadata += [meta]
            output_dispatchers += [dispatcher]
        
        return feedback_inputs, metadata, output_dispatchers
    
    def dispatch(self, control_outputs, metadata, output_dispatchers):
        """
        Dispatch control outputs back to caller(s)

        Parameters
        ----------
        control_outputs : List of 1D numpy array
            Control outputs to dispatch back.
        metadata : List of objects
            List of metadata objects. These are not used by the server 
            but simply passed back to "client" via the dispatcher for reference.
            They can be anything
        output_dispatchers : List
            List of callable. These will be called to dispatch control output
            results.

        Returns
        -------
        None.

        """
        
        # Pop out outputs, use dispatcher object to send back outputs:
        while len(control_outputs) > 0:
            output = control_outputs.pop()
            meta = metadata.pop()
            dispatcher = output_dispatchers.pop()
            dispatcher(output, meta)
    
    def run_feedback(self, feedback_inputs):
        """
        Run feedback on list of feedback inputs, and return splitted outputs.
        
        The point of this method is to process all feedback inputs in one big
        batch instead of calling feedback(), and therefore predict(), in 
        multiple small batches.

        Parameters
        ----------
        feedback_inputs : List
            List of inputs and objectives to pass to the controller's
            feedback() method.

        Returns
        -------
        strategies : List of 1D numpy arrays
            Control strategies/outputs split in the same way as the feedback 
            inputs.

        """
        
        if len(feedback_inputs) == 0:
            return []
        
        # Compile inputs & objectives arrays:
        inputs = np.concatenate([x[0] for x in feedback_inputs],axis=0)
        objectives = np.concatenate([x[1] for x in feedback_inputs],axis=0)
        
        
        t_start = time.perf_counter()
        
        # Run feedback control:
        strategies = self.controller.feedback(inputs, objectives)
        
        self._msg(
            f"{objectives.shape[0]} inputs processed ({time.perf_counter() -t_start:.2f}s)"
            )
        
        # Split strategies back to inputs dimensions:
        strategies = np.split(
            strategies,
            np.cumsum([len(x[0]) for x in feedback_inputs[:-1]]),
            axis = 0
            )
        
        return strategies

class SocketServer(Thread):
    
    def __init__(self, control_server, port = 7555, savelog = None, name = "SocketServer_1"):
        """
        Instanciate

        Parameters
        ----------
        control_server : TYPE
            DESCRIPTION.

        Returns
        -------
        None.

        """
        
        # Init Thread:
        super().__init__(daemon=True)
        
        # Copy Server ref:
        self.control_server = control_server
        
        # Start online server:
        self.port = port
        self.server_socket = socket.socket() 
        self.server_socket.bind(('',self.port))
        self.server_socket.listen(1)
        
        # Logging:
        self.name = name
        self.savelog = savelog
        if self.savelog is not None:
            os.makedirs(self.savelog, exist_ok=True)
            # self.logger = Logger(os.path.join(self.savelog,"stdout.log"))
        
        # Misc:
        self._stop_flag = False
        
        # Print init message:
        msg = f"""Initialized server
            hostname: {socket.gethostname()}
            IP adress: {socket.gethostbyname(socket.gethostname())}
            Port: {self.port}
            log folder: {self.savelog}
            
            """
        self._msg(msg)
    
    def _msg(self, msg):
        print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} {self.name}: {msg}")
    
    def run(self):
        
        self._msg("Started online server")
        
        while not self._stop_flag:
            time.sleep(.05)
            # Check if new client connection:
            client_connection,client_address=self.server_socket.accept()
            self._msg(f" New Connection: {client_address}")
            cc = ClientConnection(client_connection, self.control_server)
            cc.start()
        
        self.server_socket.close()
        self._msg("Stopped online server")
        if self.logger is not None:
            time.sleep(1) # probably not necessary
            self.logger.close()
    
    def stop(self):
        self._stop_flag = True

class ClientConnection(Thread):
    
    def __init__(self, connection, control_server):
        
        super().__init__(daemon=True)
        
        self.connection = connection
        self.connection.setblocking(0)
        self.control_server = control_server
    
    def run(self):
        
        # Wait to receive data:
        inputs, meta = self.recv()
        print(meta)
        
        # Send to controller queue:
        self.control_server.queue.put(
            (inputs, meta, self.dispatch)
            )
    
    def recv(self):
        
        countdown = 50
        ultimate_buffer=b''
        while True:
            
            # Retrieve data to temporary buffer:
            try:
                receiving_buffer = self.connection.recv(1024)
            
            # In some cases, this is raised when no data is available:
            except Exception as e:
                ex_type, ex_value, ex_traceback = sys.exc_info()
                if ex_type.__name__ == "BlockingIOError":
                    time.sleep(.01)
                    if len(ultimate_buffer)>0:
                        countdown-=1
                        if countdown<=0:
                            break
                    continue
            
            # Check if we have reached end of data:
            if len(receiving_buffer)==0:
                if len(ultimate_buffer)>0:
                    countdown-=1
                    if countdown<=0:
                        break
                else: # No data received yet
                    time.sleep(.01)
                    continue
            
            ultimate_buffer+= receiving_buffer

        # De-serialize:
        inputs, meta = pickle.loads(ultimate_buffer)
        
        return inputs, meta
    
    def dispatch(self, output, meta):
        
        # Serialize and send:
        sendback = pickle.dumps([output, meta])
        self.connection.sendall(sendback)
        self.connection.close()

class DistantServer(Thread):
    
    def __init__(self, server_address, port = 7555, fallback_server = None):
         
        super().__init__(daemon=True)
        
        self.server_address = server_address
        "Address of the remote server (SocketServer)"
        self.port = port
        "Port the SocketServer is listening to"
        self.open_sockets = []
        "List of currently open sockets"
        self.queue = Queue()
        "Queue to receive control inputs"
        self.fallback = fallback_server
        "Local or remote fallback server in case connection fails with this one"
    
    def run(self):
        
        while True:
            time.sleep(.1)
            self._run()

    def _run(self):
        """
        Function that performs the actual run operations.

        Returns
        -------
        None.

        """
        
        
        # Retrieve inputs from queue:
        while not self.queue.empty():
            feedback_inputs, metadata, output_dispatcher = self.queue.get()
                
            # Send out to controller
            try:
                connection = self.send(feedback_inputs, metadata, output_dispatcher)
            except:
                time.sleep(.5)
                try:
                    # Try once more:
                    connection = self.send(
                        feedback_inputs, metadata, output_dispatcher
                        )
                except Exception as e:
                    if self.fallback is not None:
                        print("Caught Exception below trying to send, falling back...")
                        traceback.print_exc()
                        self.fallback.queue.put(
                            (feedback_inputs, metadata, output_dispatcher)
                            )
                    else:
                        print("Caught Exception below trying to send, dispatching Falses...")
                        traceback.print_exc()
                        output_dispatcher([False]*len(feedback_inputs[0]), metadata)
                    continue
            
            # Add to open sockets list:
            self.open_sockets.append([connection, feedback_inputs, metadata, output_dispatcher])
        
        # Run through open sockets and see if they have data available :
        for sock in self.open_sockets:
            
            # Expand:
            connection, feedback_input, metadata, output_dispatcher = sock
            
            # Socket has already been closed (by remote?)
            if connection._closed:
                continue
            
            # See if control data has been returned:
            try:
                output = self.recv(connection)
            except Exception as e:
                if self.fallback is not None:
                    print("Caught Exception below trying to recv, falling back...")
                    traceback.print_exc()
                    self.fallback.queue.put(
                        feedback_input, metadata, output_dispatcher
                        )
                    sock[0] = FakeSocket()
                    continue
                else:
                    print("Caught Exception below trying to recv, dispatching Falses...")
                    traceback.print_exc()
                    output = ([False]*len(feedback_input[0]), metadata)
            
            # Data has not been returned yet
            if output is None:
                continue
        
            # Dispatch:
            output_dispatcher(*output)
        
        # Get rid of closed sockets:
        self.open_sockets = [s for s in self.open_sockets if not s[0]._closed]
        
    
    def send(self, feedback_inputs, metadata, output_dispatcher):
        
        # Create new connection to server:
        try:
            client_socket=socket.socket()
            client_socket.connect((self.server_address, self.port))
        except Exception:
            time.sleep(.5)
            try: # Try once more:
                print("Caught Exception below, trying connect once more...")
                traceback.print_exc()
                client_socket=socket.socket()
                client_socket.connect((self.server_address, self.port))
            except Exception:
                if self.fallback is not None:
                    print("Caught exception again, falling back to local server")
                    traceback.print_exc()
                    self.fallback.queue.put(
                        (feedback_inputs, metadata, output_dispatcher)
                        )
                    return FakeSocket()
        
        # Serialize data and send it:
        data_b = pickle.dumps([feedback_inputs, metadata])
        client_socket.sendall(data_b)
        
        return client_socket
    
    def recv(self, connection):
        
        # Set non-blocking so we can return None if not done:
        connection.setblocking(0)
        ultimate_buffer=b''
        countdown = 10
        while True:
            
            # Retrieve data to temporary buffer:
            try:
                receiving_buffer = connection.recv(1024)
            
            # In some cases, this is raised when no data is available:
            except Exception as e:
                ex_type, ex_value, ex_traceback = sys.exc_info()
                if ex_type.__name__ == "BlockingIOError" :
                    if len(ultimate_buffer)>0:
                        time.sleep(.01)
                        countdown-=1
                        if countdown<=0:
                            break
                    else: # No data received yet
                        return None
                else:
                    raise e
            
            # Check if we have reached end of data:
            if len(receiving_buffer)==0:
                if len(ultimate_buffer)>0:
                    time.sleep(.01)
                    countdown-=1
                    if countdown<=0:
                        break
                else: # No data received yet
                    return None
            
            # Append to larger buffer:
            ultimate_buffer+= receiving_buffer
        
        # De-serialize data:
        control_output, metadata = pickle.loads(ultimate_buffer)
        
        # Make sure socket is deallocated and closed:
        connection.shutdown(socket.SHUT_RDWR)
        connection.close()
        
        return control_output, metadata

class FakeSocket():
    _closed = True


class Logger(object):
    # https://stackoverflow.com/questions/14906764/how-to-redirect-stdout-to-both-file-and-console-with-scripting
    def __init__(self, logfile):
        self.terminal = sys.stdout
        self.log = open(logfile, "w")
        sys.stdout = self
   
    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)
    
    def close(self):
        sys.stdout = self.terminal
        self.log.close()

    def flush(self):
        # this flush method is needed for python 3 compatibility.
        # this handles the flush command by doing nothing.
        # you might want to specify some extra behavior here.
        pass    