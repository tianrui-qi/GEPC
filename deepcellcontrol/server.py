# -*- coding: utf-8 -*-
"""
Created on Mon Dec  6 14:20:34 2021

@author: jeanbaptiste
"""

import time
from threading import Thread
from queue import Queue

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
        with tf.device(self.device):
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
