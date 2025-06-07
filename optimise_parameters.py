from sklearn.model_selection import KFold

def optimise_parameters(x_data,y_data,x_test,y_test,
                    space_model,final_activation,
                    n_runs,num_op,num_initial,
                    n_splits=5,max_params_ratio = 5.0):
    
    from skopt import gp_minimize
    from sklearn.metrics import r2_score
    from build_model import build_model
    import numpy as np
    from tensorflow.keras import backend as K
    import gc
    
    from tensorflow.keras.callbacks import EarlyStopping
    import skopt.utils as sku

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
    
    # Plots a graph of the minimisation progress
    #def plot_gp_progress(all_runs):
    #    plt.rcParams.update({'font.size': 14})
    #    fig, ax = plt.subplots(figsize=(10, 6))
    #    
    #    for i, result in enumerate(all_runs):
    #        ax.plot(result.func_vals, label=f'Run {i+1}')
    #    
    #    ax.set_xlabel('Iteration')
    #    ax.set_ylabel('Objective Value')
    #    ax.set_yscale('log')
    #    ax.set_title('GP Minimization Progress Across Runs')
    #    ax.legend()
    #    ax.grid(True)
    #    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    #    filename = f"plots/gp_mini_{timestamp}.png"
    #    os.makedirs(os.path.dirname(filename), exist_ok=True)
    #    fig.savefig(filename, dpi=600, bbox_inches='tight')
    #    print(f"Plot saved to {filename}")
    #    plt.show()
    #    plt.close()
    
    #Builds a layered model, that can be tapered, with a minimum of 8 units per layer.
    def tapered_layers(base_units, taper_rate, num_layers=3, min_units=8):
        return [max(min_units, int(base_units // (taper_rate ** i))) for i in range(num_layers)]
    
    def print_progress(res):
        print(f"[{len(res.x_iters)}/{num_op}] Step completed.")
    
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    @sku.use_named_args(dimensions=space_model)
    def objective(lr, epochs, layers, hidden_units, taper_rate, dropout_rate, l2_factor):
        val_losses = []
        val_r2s = []
        try:
            for train_idx, val_idx in kf.split(x_data):
                # Clear previous session to free memory
                K.clear_session()
                
                x_train, x_valid = x_data[train_idx], x_data[val_idx]
                y_train, y_valid = y_data[train_idx], y_data[val_idx]
                
                
                model = build_model(
                    input_dim=x_train.shape[1],
                    hidden_units = tapered_layers(hidden_units, taper_rate, layers),
                    #activation='relu',
                    lr=lr,
                    use_adjusted_softmax=use_adjusted_softmax,
                    use_threshold_activation=use_threshold_activation,
                    use_normalized_relu=use_normalized_relu,
                    final_activation=final_activation,
                    dropout_rate=dropout_rate,
                    l2_factor=l2_factor,
                    output_width=y_train.shape[1]
                )
                
                if model.count_params() > x_train.shape[0] * max_params_ratio:
                    print(f"⚠️ Skipped: {model.count_params()} params > {int(x_train.shape[0] * max_params_ratio)} allowed.")
                    val_loss = 1e6
                    r2 = -10
                else:
                    # Train the model using the training set and evaluate on the validation set.
                    history_optimise = model.fit(
                        x_train, y_train,
                        validation_data=(x_valid, y_valid),
                        epochs=epochs,
                        batch_size=32,
                        callbacks=[early_stop],
                        verbose=0 # set to 0 for tuning so that training output isn't printed each time
                    )
                    
                    # Evaluate the model on the validation set (taking an average on the last 10 points).
                    val_loss = np.mean(history_optimise.history['val_loss'][-20:])
                    
                    # Compute R^2 on the validation set.
                    y_pred = model.predict(x_valid)
                    r2 = r2_score(y_valid,y_pred)
                
                val_losses.append(val_loss)
                val_r2s.append(r2)
            
            avg_val_loss = np.mean(val_losses)
            avg_r2 = np.mean(val_r2s)
            
            # Set a lambda (penalty weight) for combining metrics.
            lambda_r2 = 2.0 # Weight on the (1-R^2) term
            r2_threshold=0.85
            penalty_r2 = lambda_r2 * max(0.0,(r2_threshold - avg_r2))
            
            # Overall objective: lower is better, we want low loss and high R2.
            objective_value = avg_val_loss + penalty_r2 #+ penalty_diff + penalty_var
            
            print(f"Evaluated: lr = {lr:.2e}, epochs = {epochs}, layers = {layers}, hidden units = {hidden_units},")
            print(f"           taper rate = {taper_rate:.3f}, dropout rate = {dropout_rate:.3f}")
            print(f"    => val loss = {avg_val_loss:.4e}, R2 = {avg_r2:.4f}, objective = {objective_value:.4e}")
            
            # Return the validation loss as the objective.
            return objective_value
            
        except Exception as e:
            print(f"⚠️  Error during trial with lr={lr:.1e}, epochs={epochs}, hidden_units={hidden_units}: {e}")
            return 1e6  # Large penalty to avoid choosing this combination
        
        finally:
            # Clean up to reduce memory usage
            try:
                del model
                del history_optimise
            except NameError:
                pass
            gc.collect()
            K.clear_session()
    
    
    res_out = None
    best_fun = float("inf")
    for i in range(n_runs):
        print(f"\nRunning optimization {i+1}/{n_runs}...\n")
        
        # Clear memory
        K.clear_session()
        gc.collect()
        
        result = gp_minimize(
            func=objective,
            dimensions=space_model,
            n_initial_points=num_initial,
            n_calls=num_op,
            acq_func='gp_hedge',#EI or LCB
            noise="gaussian",#1e-6,
            random_state=42+i,
            callback=[print_progress]
        )
        
        # for the cluseter, memory saving to just save the best run not all the runs.
        if result.fun < best_fun:
            res_out = result
            best_fun = result.fun
    
    best_lr, best_epochs, best_layers, best_hidden_units, best_taper_rate, best_dropout, best_l2 = res_out.x
    print(f"Evaluated: lr = {best_lr:.2e}, epochs = {best_epochs}, layers = {best_layers}, hidden units = {best_hidden_units},")
    print(f"           taper rate = {best_taper_rate:.3f}, dropout rate = {best_dropout:.3f}, l2 = {best_l2:.3e}")
    
    with open("best_hyperparams.txt", "a") as f:
        f.write(f"Evaluated: lr = {best_lr}, epochs = {best_epochs}, layers = {best_layers}, hidden units = {best_hidden_units},\n")
        f.write(f"           taper rate = {best_taper_rate}, dropout rate = {best_dropout}, l2 = {best_l2}\n")
    
    # Clear the model
    K.clear_session()
    gc.collect()
    
    return res_out