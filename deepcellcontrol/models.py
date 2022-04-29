#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 14 18:24:45 2020

@author: jeanbaptiste
"""
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, LSTM, Input, Dropout, TimeDistributed
from tensorflow.keras.optimizers import Adam
# from tensorflow.autograph.experimental import do_not_convert
import tensorflow as tf
from tensorflow import keras

def lstm(
        past_steps=36, 
        horizon=24,
        features_dim=2,
        latent_dim=200,
        output_dim = 1,
        activation='linear',
        loss='mse',
        metrics=None,
        learning_rate=0.001
        ):
    
    # Inputs:
    past_events = Input((past_steps,features_dim),name='past_inputs') # Past fluo + light inputs
    future_light = Input((horizon,1),name='future_inputs') # Only future light inputs
    
    # Encoding LSTM: (return internal states to feed into decoder)
    _, state_h, state_c = LSTM(
        latent_dim,return_state=True,name='encoder'
        )(past_events)
    
    # Decoding LSTM: (Initialize with decoder states, pass future light inputs)
    decoder_outputs = LSTM(
        latent_dim, return_sequences=True, name='decoder',
        )(future_light,initial_state=[state_h, state_c])
    
    # Compile prediction over horizon:
    prediction = TimeDistributed(
        Dense(output_dim,activation=activation,name='prediction')
        )(decoder_outputs)

    # Finalize model:
    model = Model([past_events, future_light], prediction)
    model.compile(
        loss=loss, optimizer=Adam(learning_rate=learning_rate),metrics=metrics
        )
    
    return model
  

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