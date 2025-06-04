def train_model(x_data,y_data,x_test,y_test,
                    final_activation,res_out):
    
    from sklearn.metrics import mean_absolute_error
    from sklearn.metrics import r2_score
    from build_model import build_model
    
    from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping

    lr_scheduler = ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=10,
        min_lr=1e-7,
        verbose=1
    )
    
    early_stop = EarlyStopping(
        monitor='val_loss',
        patience=20,              # Stop after 20 epochs with no improvement
        restore_best_weights=True, # Restore best weights after stopping
        verbose=0
    )
    
    use_adjusted_softmax=False
    use_threshold_activation=False
    use_normalized_relu=False
    if final_activation == 'adjusted_softmax':
        use_adjusted_softmax=True
    elif final_activation == 'threshold_activation':
        use_threshold_activation=True
    elif final_activation == 'normalized_relu':
        use_normalized_relu=True
    
    #Builds a layered model, that can be tapered, with a minimum of 8 units per layer.
    def tapered_layers(base_units, taper_rate, num_layers=3, min_units=8):
        return [max(min_units, int(base_units // (taper_rate ** i))) for i in range(num_layers)]
    
    # Now once we have the best hyperparameters, we can rebuild the final model with these values
    best_lr, best_epochs, best_layers, best_hidden_units, best_taper_rate, best_dropout, best_l2 = res_out.x
    
    # Buld the final model using the best hyperparameters:
    final_model = build_model(
        input_dim=x_data.shape[1],
        hidden_units = tapered_layers(best_hidden_units, best_taper_rate, best_layers),
        #activation='relu',
        lr=best_lr,
        use_adjusted_softmax=use_adjusted_softmax,
        use_threshold_activation=use_threshold_activation,
        use_normalized_relu=use_normalized_relu,
        final_activation=final_activation,
        dropout_rate=best_dropout,
        l2_factor=best_l2,
        output_width=y_data.shape[1]
    )
    
    history_final = final_model.fit(
        x_data, y_data,
        validation_split=0.1,
        epochs=best_epochs,
        batch_size=32,
        callbacks=[lr_scheduler],
        verbose=1
    )
    
    # Finally, evaluate on the untouched test set:
    y_pred_test = final_model.predict(x_test)
    final_mae = mean_absolute_error(y_test, y_pred_test)
    final_r2 = r2_score(y_test, y_pred_test)
    print(f"Final Test MAE: {final_mae:.4f}, R2: {final_r2:.4f}")
    
    return final_model, history_final