
function sigmoid(z)
  # sigmoid is a basic sigmoid function returning values from 0-1
  1. / ( 1. + 1e.^(-z) )
end

function sigmoidGradient(z)
  sigmoid(z) .* ( 1 - sigmoid(z) )
end

function initialize_theta(input_unit_count, output_class_count, hidden_unit_length_list)
  #
  #initialize_theta creates architecture of neural network
  #    
  #Parameters:
  #  hidden_unit_length_list - Array of hidden layer units
  #  input_unit_count - integer, number of input units (features)
  #  output_class_count - integer, number of output classes
  #
  #Returns:
  #  
  #Array of theta arrays randomly initialized to from -.5 to .5
  #
  
  if length( hidden_unit_length_list ) == 0
    hidden_unit_length_list = [2]
  end
  
  unit_count_list = [input_unit_count]
  unit_count_list = [unit_count_list, hidden_unit_length_list]
  unit_count_list = [unit_count_list, output_class_count]
  layers = length(unit_count_list)
  
  Theta_L = [ rand( unit_count_list[i], unit_count_list[i-1]+1 ) - .5 for i = 2:layers]

end

function print_theta(Theta_L)
  # print_theta() is a helper function that prints Theta_L and architecture info
  # It does not actually "do" anything except print to stdout

  T = length(Theta_L)

  println()
  println( "NN ARCHITECTURE" )
  println( "$(T+1) Layers ($(T-1) Hidden)" )
  println( "$T Thetas" )
  println( "$(size(Theta_L[1],2)-1) Input Features" )
  println( "$(size(Theta_L[end], 1)) Output Classes" )
  println()
    
  println( "Units per layer (excl. bias unit)" )
  for t = 1:T
    if t == 1
      println( " - Input: $(size(Theta_L[t],2)-1) Units" )
    end
    if t < T
      println( " - Hidden $t: $(size(Theta_L[t],1)) Units" )
    else
      println( " - Output: $(size(Theta_L[t],1)) Units" )
    end
  end
  println()

  println( "Theta Shapes" )
  for l = 1:T
    println( "Theta $l: $(size(Theta_L[l]))" )
  end
  println()
  
  println( "Theta Values" )
  for t= 1:T
    println( "Theta $t:"  )
    println ( Theta_L[t])
  end
  println()

end

function nn_cost(Y, Y_pred)
  #
  # nn_cost implements cost function for array inputs Y and Y_pred
  #  
  # y is array of n_observations by n_classes
  # Y_pred is array with same dimensions as Y of predicted y values
  #
  if size(Y) != size(Y_pred)
    if size(Y,1) != size(Y_pred,1)
      error("Wrong number of predictions", "$(size(Y,1)) Actual Values. $(size(Y_pred,1)) Predicted Values. ")
    else
      error("Wrong number of prediction classes", "$(size(Y,2)) Actual Classes. $(size(Y_pred,2)) Predicted Classes. ")
    end
  end
    
  n_observations = size(Y,1)
  
  # Cost Function
  if size(Y,2) == 1
    # Y[i] is scalar
    J = (-1.0 / n_observations ) * sum(Y'*log(Y_pred) + ((1-Y')*log(1-Y_pred)))
  else
    # Y[i] is vector
    J = (-1.0 / n_observations ) * sum(diag(Y'*log(Y_pred) + ((1-Y')*log(1-Y_pred))))
  end
  
  J

end



function nn_predict(X, Theta_L)
  #
  # nn_predict calculates activations for all layers given X and thetas in Theta_L
  # return all inputs and activations for all layers for use in backprop
  #
  # Parameters
  #  X is matrix of input features dimensions n_observations by n_features
  #  Theta_L is a 3D array where first element corresponds to the layer number, second is unit at layer+1, third is unit in layer
  #
  # Returns
  #  a_N - 1D Array of activation 2D arrays for each layer: Input (1), Hidden (2 to T), and Output (T+1)
  #  a_Z - 1D Array of input 2D arrays to compute activations for all non-bias units
  #  a_N[end] - 2D Array of predicted Y values with dimensions n_observations by n_classes
  #

  a_N = {}
  z_N = {}

  m = size(X,1)
  T = length(Theta_L)
    
  # Input Layer inputs
  push(a_N, X) 			# List of activations including bias unit for non-output layers
  push(z_N, zeros(1,1)) 	# add filler Z layer to align keys/indexes for a, z, and Theta

  # Loop through each Theta_List theta
  # t is index of Theta for calculating layer t+1 from layer t
  for t=1:T
    # Reshape 1D Array into 2D Array
    if ndims(a_N[t]) == 1
      a_N[t] = reshape(a_N[t], 1, size(a_N[t],1))
    end

    # Add bias unit
    a_N[t] = [ ones(size(a_N[t],1), 1)  a_N[t] ]
      
    # Calculate and Append new z and a arrays to z_N and a_N lists
    push(z_N, a_N[t] * Theta_L[t]') #'
    push(a_N, sigmoid(z_N[t+1]))
  end
  
  z_N, a_N, a_N[end]

end


function back_prop(X_train, Y_train, Theta_L, lmda)
  #
  # Parameters
  #  X_train - Array of feature inputs with dimensions n_observations by n_features
  #  Y_train - Array of class outputs with dimensions n_observations by n_classes
  #  Theta_L is a 1D array of Theta values where 1D element is the layer number, the 2D elements are unit in layer+1 and unit in layer
  #  lmda - Float64 - lambda term for regularization
  #  
  # Returns
  #  Y_pred as array of predicted Y values from nn_predict()
  #  Theta_Gradient_L as 1D array of 2D Theta Gradient arrays
  #

  n_observations = size(X_train,1)
  
  T = length(Theta_L)

  # Create Modified copy of the Theta_L for Regularization with Coefficient for bias unit set to 0 so that bias unit is not regularized
  # Create variable to accumulate error caused by each Theta_L term in layer a_N[n+1]
  Theta_Gradient_L = {}
  regTheta = {}
  for i=1:T
    push(regTheta, [zeros(size(Theta_L[i],1),1) Theta_L[i][:,2:]])
    push(Theta_Gradient_L, zeros(size(Theta_L[i])))
  end


  # Forward Pass
  z_N, a_N, Y_pred = nn_predict(X_train, Theta_L)

  # Backprop Error Accumulator
  delta_N = {}
  for t=1:T
    push(delta_N, {})
  end
    
  # Error for Output layer is predicted value - Y training value
  delta = Y_pred - Y_train
  if ndims(delta) == 1
    delta = reshape(delta, 1, length(delta) )
  end

  # Loop backwards through Thetas to apply Error to prior Layer (except input layer)
  # Finish at T-2 because start at 0, output layer is done outside, the loop and input has no error

  # Output Error
  delta_N[T] = delta

  # Hidden Layers Error    
  for t=0:T-2
    delta = (delta * Theta_L[T-t][:,2:]) .* sigmoidGradient(z_N[T-t])
    delta_N[T-t-1] = delta
  end
    
  # Calculate error gradients (no error in input layer)
  # t is the Theta from layer t to layer t+1
  for t=1:T
    Theta_Gradient_L[t] = Theta_Gradient_L[t] + delta_N[t]' * a_N[t] #'
  end

  # Average Error + regularization penalty  
  for t=1:T
    Theta_Gradient_L[t] = Theta_Gradient_L[t] * (1.0/n_observations) + (lmda * regTheta[t])
  end
  
  Y_pred, Theta_Gradient_L

end



function fit(X_train, Y_train, Theta_L, lmda, epochs)
  #
  #fit() calls the training back_prop function for the given number of cycles
  #tracks error and error improvement rates
  #  
  #Parameters:
  #  X_train - Array of training data with dimension n_observations by n_features
  #  Y_train - Array of training classes with dimension n_observations by n_classes
  #  Theta_L - 1D array of theta 2d arrays where each theta has dimensions n_units[layer+1] by n_units[layer]+1
  #  epochs -  integer of number of times to update Theta_L
  #  
  #Returns
  #  Theta_L - 1D array of Theta arrays
  #  J_List - Array (length = epochs) of result of cost function for each iteration

  J_list = zeros( epochs )

  for i=1:epochs
    # Back prop to get Y_pred and Theta gradient
    Y_pred, Theta_grad = back_prop(X_train, Y_train, Theta_L, lmda)
    # Record cost
    J_list[i] = nn_cost(Y_train, Y_pred)
    # Update Theta using Learning Rate * Theta Gradient
    for t=1:length(Theta_L)
      # Start with a large learning rate; need to update this to be more intelligent than simply looking at iteration count
      # Need to update to change learning rate based on progress of cost function
      if i < 100
        learning_rate = 5.0        
      else
        learning_rate = 1.0
      end
      Theta_L[t] = Theta_L[t] - ( learning_rate * Theta_grad[t] )
    end
    #println("Cost $i: $(J_list[i])")
  end

  Theta_L, J_list

end


function XOR_test(hidden_unit_length_list, epochs)
  #
  #XOR_test is a simple test of the nn printing the predicted value to std out
  #Trains on a sample XOR data set
  #Predicts a single value
  #Accepts an option parameter to set architecture of hidden layers
  #
  
  
  println( "Training Data: X & Y")

  # Training Data
  X_train = [1 1; 1 0; 0 1; 0 0]	# Training Input Data
  Y_train = [0 1; 1 0; 1 0; 0 1] 			# Training Classes
  println( X_train )
  println( Y_train )
  
  # Hidden layer architecture
  hidden_layer_architecture = hidden_unit_length_list

  # Regularization Term
  lmda = 1e-5

  # Initialize Theta based on selected architecture
  Theta_L = initialize_theta(size(X_train,2), size(Y_train,2), hidden_layer_architecture)

  
  # Fit
  Theta_L, J_list = fit(X_train, Y_train, Theta_L, lmda, epochs)
  
  # Print Architecture
  print_theta(Theta_L)
  
  # Print Cost
  println("Cost Function Applied to Training Data: $(J_list[end])")

  # Predict
  X_new = [1 0;0 1;1 1;0 0]
  println( "Given X: $X_new")
  z_N, a_N, Y_pred = nn_predict(X_new, Theta_L)
  println( "Predicted Y: $(round(Y_pred,3))")

  Y_pred

end



Y_pred = XOR_test([2], 5000)
