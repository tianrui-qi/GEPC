#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 18:24:45 2020

@author: jeanbaptiste
"""
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Input, Dropout, TimeDistributed
from tensorflow.keras.optimizers import Adam

def lstm(hyper_parameters):
    
    # Inputs:
    past_events = Input(
        (hyper_parameters["past_steps"], len(hyper_parameters["features"])),
        name='past_inputs'
        )
    future_light = Input((None,1),name='future_inputs')
    
    # Encoder:
    state_h, state_c = _encoder(past_events, hyper_parameters)
    
    # Decoder:
    prediction = _decoder(state_h, state_c, future_light, hyper_parameters)

    # Finalize model:
    model = Model([past_events, future_light], prediction)
    model.compile(
        loss=hyper_parameters["loss"],
        optimizer = Adam(
            learning_rate=hyper_parameters["learning_rate"]
            )
        )
    
    return model

def lstm_encoder(hyper_parameters):
    
    # Inputs:
    past_events = Input(
        (hyper_parameters["past_steps"], len(hyper_parameters["features"])),
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

def lstm_decoder(hyper_parameters):
    
    # Inputs:
    state_h = Input((hyper_parameters["latent_dim"],),name='state_h') 
    state_c = Input((hyper_parameters["latent_dim"],),name='state_c') 
    future_light = Input((None,1),name='future_inputs')
  
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

def _encoder(past_events,hyper_parameters):
    
    # Encoding LSTM: (return internal states to feed into decoder)
    intermediate = LSTM(64,name='encoder_1',return_sequences=True)(past_events)
    _, state_h, state_c = LSTM(
        hyper_parameters["latent_dim"],return_state=True,name='encoder_2'
        )(intermediate)
    
    return state_h, state_c


def _decoder(state_h, state_c, future_light, hyper_parameters):
    
    
    if hyper_parameters["output_mode"]=="timedistributed":
        # Decoding LSTM: (Initialize with decoder states, pass future light inputs)
        decoder_outputs = LSTM(
            hyper_parameters["latent_dim"],
            return_sequences=True,
            name='decoder_1',
            )(future_light,initial_state=[state_h, state_c])
        # Compile prediction over horizon:
        prediction = TimeDistributed(
            Dense(hyper_parameters["output_dim"],activation="linear"),
            name="decoder_2"
            )(decoder_outputs)
    elif hyper_parameters["output_mode"]=="dense":
        decoder_outputs = LSTM(
            hyper_parameters["latent_dim"], name='decoder_1',
            )(future_light,initial_state=[state_h, state_c])
        prediction = Dense(
            hyper_parameters["output_dim"],activation="linear",name="decoder_2"
            )(decoder_outputs)
    
    return prediction
    

def split(model):
    
    # Collect hyper_parameters:
    hyper_parameters = dict(
        past_steps = model.get_layer("past_inputs").output_shape[0][1],
        features = ["" for _ in range(model.get_layer("past_inputs").output_shape[0][2])], # dummy list to for the len() call
        latent_dim = model.get_layer("decoder_1").output_shape[-1],
        output_mode = model.get_layer("decoder_2").__class__.__name__.lower(),
        output_dim = model.get_layer("decoder_2").output_shape[-1],
        loss = model.loss, # These aren' really useful
        learning_rate = float(model.optimizer.learning_rate), # These aren' really useful
        )
    
    # Create encoder and transfer weights from model:
    encoder = lstm_encoder(hyper_parameters)
    _transfer(model, encoder, "encoder_1")
    _transfer(model, encoder, "encoder_2")
    
    # Create decoder and transfer weights from model:
    decoder = lstm_decoder(hyper_parameters)
    _transfer(model, decoder, "decoder_1")
    _transfer(model, decoder, "decoder_2")
    
    return encoder, decoder


def _transfer(model1, model2, layer_name):
    weights = model1.get_layer(layer_name).get_weights()
    model2.get_layer(layer_name).set_weights(weights)

# TODO remove?
def mlp(
        past_steps=36,
        features=2,
        horizon=24,
        hidden_layers=10, 
        hidden_dim=64,
        dropout=0,
        activation='linear',
        loss='mse',
        metrics=None,
        learning_rate=0.001
        ):
    
    model = Sequential()
    model.add(Input(shape=(features*past_steps + horizon,)))
    for _ in range(hidden_layers):
        model.add(Dense(hidden_dim,activation='relu'))
        if dropout > 0:
            model.add(Dropout(dropout))
    model.add(Dense(horizon,activation=activation))
    model.compile(loss=loss, optimizer=Adam(learning_rate=learning_rate),metrics=metrics)
    
    return model