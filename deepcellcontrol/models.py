#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Model definitions for timeseries prediction + architecture manipulation utils

Created on Fri Aug 14 18:24:45 2020

@author: jeanbaptiste
"""
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, LSTM, Input, TimeDistributed, Concatenate
from tensorflow.keras.optimizers import Adam


def lstm_mlp(hyper_parameters):
    """
    Get LSTM-MLP encoder-decoder network

    Parameters
    ----------
    hyper_parameters : Dict
        Hyper-parameters dictionary. See config.defaults for a list of 
        parameters

    Returns
    -------
    model : Model
        Compiled LSTM-MLP model.

    """
    
    # Inputs:
    past_events = Input(
        (None, len(hyper_parameters["features"])),
        name='past_inputs'
        )
    future_light = Input((hyper_parameters["horizon"],),name='future_inputs')
    
    # Encoder:
    state_h, state_c = _encoder(past_events, hyper_parameters)
    
    # Decoder:
    prediction = _mlpdecoder(state_h, state_c, future_light, hyper_parameters)

    # Finalize model:
    model = Model([past_events, future_light], prediction)
    model.compile(
        loss=hyper_parameters["loss"],
        optimizer = Adam(
            learning_rate=hyper_parameters["learning_rate"]
            )
        )
    
    return model


def split(model, decode_mode="mlp"):
    """
    Split encoder-decoder model into encoder and decoder parts.

    Parameters
    ----------
    model : Model
        Encoder-decoder model.
    decode_mode : str, optional
        Decoder architecture type, options are "lstm" or "mlp".
        The default is "mlp".

    Returns
    -------
    encoder : Model
        Encoder model (not compiled).
    decoder : Model
        Decoder model (not compiled).

    """
    
    # Collect hyper_parameters:
    hyper_parameters = dict(
        past_steps = model.get_layer("past_inputs").output_shape[0][1],
        features = ["" for _ in range(model.get_layer("past_inputs").output_shape[0][2])], # dummy list to for the len() call
        latent_dim = model.get_layer("encoder_1").output_shape[0][-1],
        output_mode = model.get_layer("decoder_1").__class__.__name__.lower(), # Only useful for lstm decoder
        loss = model.loss, # These aren' really useful
        learning_rate = float(model.optimizer.learning_rate), # These aren' really useful
        )
    if decode_mode == "mlp":
        hyper_parameters["mlp_layers"] = -1
        hyper_parameters["mlp_dim"] = model.get_layer("decoder_0").output_shape[-1]
        for layer in model.layers:
            if layer.name.startswith("decoder_"):
                hyper_parameters["mlp_layers"]+=1
                hyper_parameters["horizon"] = layer.output_shape[-1]
            
    
    # Create encoder and transfer weights from model:
    encoder = lstm_encoder(hyper_parameters)
    _transfer(model, encoder, "encoder_0")
    _transfer(model, encoder, "encoder_1")
    
    # Create decoder and transfer weights from model:
    if decode_mode == "lstm":
        decoder = lstm_decoder(hyper_parameters)
    elif decode_mode == "mlp":
        decoder = mlp_decoder(hyper_parameters)
    for layer in model.layers:
        if layer.name.startswith("decoder_"):
            _transfer(model, decoder, layer.name)
        
    
    return encoder, decoder


def _transfer(model1, model2, layer_name):
    """
    Transfer layer weights between models

    Parameters
    ----------
    model1 : Model
        Model to copy weights from.
    model2 : Model
        Model to transfer weights to.
    layer_name : str
        Layer name. The layer name must be shared between both models.

    Returns
    -------
    None.

    """
    
    weights = model1.get_layer(layer_name).get_weights()
    model2.get_layer(layer_name).set_weights(weights)


def lstm_encoder(hyper_parameters):
    """
    Get LSTM encoder network 

    Parameters
    ----------
    hyper_parameters : Dict
        Hyper-parameters dictionary. See config.defaults for a list of 
        parameters

    Returns
    -------
    model : Model
        Compiled LSTM encoder model.

    """
    
    # Inputs:
    past_events = Input(
        (None, len(hyper_parameters["features"])),
        name='past_inputs'
        )
    
    # Encoder:
    state_h, state_c = _encoder(past_events, hyper_parameters)

    # Finalize model:
    model = Model(past_events, [state_h, state_c])
    model.compile(
        loss=hyper_parameters["loss"],
        optimizer = Adam(
            learning_rate=hyper_parameters["learning_rate"]
            )
        )
    
    return model


def _encoder(past_events,hyper_parameters):
    """
    LSTM encoder definition

    Parameters
    ----------
    past_events : tf.Tensor
        Input tensor of past timeseries. See lstm_encoder()
    hyper_parameters : Dict
        Hyper-parameters dictionary. See config.defaults for a list of 
        parameters

    Returns
    -------
    state_h : tf.Tensor
        LSTM hidden state.
    state_c : tf.Tensor
        LSTM cell state.

    """
    
    # Encoding LSTM: (return internal states to feed into decoder)
    intermediate = LSTM(64,name='encoder_0',return_sequences=True)(past_events)
    _, state_h, state_c = LSTM(
        hyper_parameters["latent_dim"],return_state=True,name='encoder_1'
        )(intermediate)
    
    return state_h, state_c


def lstm_decoder(hyper_parameters):
    """
    Get LSTM decoder network (obsolete)

    Parameters
    ----------
    hyper_parameters : Dict
        Hyper-parameters dictionary. See config.defaults for a list of 
        parameters

    Returns
    -------
    model : Model
        Compiled LSTM decoder model.

    """
    
    # Inputs:
    state_h = Input((hyper_parameters["latent_dim"],),name='state_h') 
    state_c = Input((hyper_parameters["latent_dim"],),name='state_c') 
    future_light = Input((None,),name='future_inputs')
  
    # Decoder:
    prediction = _decoder(state_h, state_c, future_light, hyper_parameters)

    # Finalize model:
    model = Model([state_h, state_c, future_light], prediction)
    model.compile(
        loss=hyper_parameters["loss"],
        optimizer = Adam(
            learning_rate=hyper_parameters["learning_rate"]
            )
        )
    
    return model


def _decoder(state_h, state_c, future_light, hyper_parameters):
    """
    LSTM decoder definition

    Parameters
    ----------
    state_h : tf.Tensor
        LSTM encoder hidden state.
    state_c : tf.Tensor
        LSTM encoder cell state.
    future_light : tf.Tensor
        Input tensor of (potential) future optogenetic sequence. 
        See lstm_mlp().
    hyper_parameters : Dict
        Hyper-parameters dictionary. See config.defaults for a list of 
        parameters

    Returns
    -------
    prediction : tf.Tensor
        Future fluorescence prediction.

    """
    
    
    if hyper_parameters["output_mode"]=="timedistributed":
        # Decoding LSTM: (Initialize with decoder states, pass future light inputs)
        decoder_outputs = LSTM(
            hyper_parameters["latent_dim"],
            return_sequences=True,
            name='decoder_0',
            )(future_light,initial_state=[state_h, state_c])
        # Compile prediction over horizon:
        prediction = TimeDistributed(
            Dense(1,activation="linear"),
            name="decoder_1"
            )(decoder_outputs)
    elif hyper_parameters["output_mode"]=="dense":
        decoder_outputs = LSTM(
            hyper_parameters["latent_dim"], name='decoder_0',
            )(future_light,initial_state=[state_h, state_c])
        prediction = Dense(
            hyper_parameters["horizon"],activation="linear",name="decoder_1"
            )(decoder_outputs)
    
    return prediction


def mlp_decoder(hyper_parameters):
    """
    Get MLP decoder network

    Parameters
    ----------
    hyper_parameters : Dict
        Hyper-parameters dictionary. See config.defaults for a list of 
        parameters

    Returns
    -------
    model : Model
        Compiled MLP decoder model.

    """
    
    # Inputs:
    state_h = Input((hyper_parameters["latent_dim"],),name='state_h') 
    state_c = Input((hyper_parameters["latent_dim"],),name='state_c') 
    future_light = Input((hyper_parameters["horizon"],),name='future_inputs')
  
    # Decoder:
    prediction = _mlpdecoder(state_h, state_c, future_light, hyper_parameters)

    # Finalize model:
    model = Model([state_h, state_c, future_light], prediction)
    model.compile(
        loss=hyper_parameters["loss"],
        optimizer = Adam(
            learning_rate=hyper_parameters["learning_rate"]
            )
        )
    
    return model


def _mlpdecoder(state_h, state_c, future_light, hyper_parameters):
    """
    MLP decoder definition

    Parameters
    ----------
    state_h : tf.Tensor
        LSTM encoder hidden state.
    state_c : tf.Tensor
        LSTM encoder cell state.
    future_light : tf.Tensor
        Input tensor of (potential) future optogenetic sequence. 
        See lstm_mlp().
    hyper_parameters : Dict
        Hyper-parameters dictionary. See config.defaults for a list of 
        parameters

    Returns
    -------
    prediction : tf.Tensor
        Future fluorescence prediction.

    """
    
    # Concatenate inputs together:
    hidden = Concatenate(
        axis=-1,
        name="prediction_inputs"
        )([state_h, state_c, future_light])
    
    # Hidden layers:
    for l in range(hyper_parameters["mlp_layers"]):
        hidden = Dense(
            hyper_parameters["mlp_dim"],
            activation='relu',
            name=f"decoder_{l}"
            )(hidden)
    
    # Output layer:
    prediction = Dense(
            hyper_parameters["horizon"],
            activation="linear",
            name=f"decoder_{l+1}"
            )(hidden)
    
    return prediction


def mlp_nopast(hyper_parameters):
    """
    Get MLP network that doesn't use encoded past (for evaluation purposes)

    Parameters
    ----------
    hyper_parameters : Dict
        Hyper-parameters dictionary. See config.defaults for a list of 
        parameters

    Returns
    -------
    model : Model
        Compiled MLP model.

    """
    
    
    future_light = Input((hyper_parameters["horizon"],),name='future_inputs')
    
    # Hidden layers:
    for l in range(hyper_parameters["mlp_layers"]):
        hidden = Dense(
            hyper_parameters["mlp_dim"],
            activation='relu',
            name=f"decoder_{l}"
            )(future_light)
    
    # Output layer:
    prediction = Dense(
            hyper_parameters["horizon"],
            activation="linear",
            name=f"decoder_{l+1}"
            )(hidden)

    # Finalize model:
    model = Model(future_light, prediction)
    model.compile(
        loss=hyper_parameters["loss"],
        optimizer = Adam(
            learning_rate=hyper_parameters["learning_rate"]
            )
        )
    
    return model
