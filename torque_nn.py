import torch.nn as nn
from torch.utils.data import DataLoader

def external_torque(x):
    return 1e-3*np.float_power(np.cos(2*x),2)*np.sin(3*x)

step = np.pi*1e-4
n = np.arange(start=0, stop=2*np.pi, step=step)
func = 1e3*external_torque(n)

X, y = n, func
X_train, y_train = X, y
X_train, y_train = X_train[:,None], y_train[:,None]

Xy_train = torch.from_numpy(np.concatenate((X_train,y_train),axis=1))

# input size, input layer size, hidden layer size, output size and batch size
input_size, n_in, n_h, n_out, batch_size = 1, 10, 10, 1, int(y_train.size/20)

# Create a model
model = nn.Sequential(
    nn.Linear(input_size, n_in),
    nn.Tanh(),
    nn.Linear(n_in, n_h),
    nn.Tanh(),
    nn.Linear(n_h, n_out),
    nn.Tanh()).double()

# Construct the loss function
criterion = torch.nn.MSELoss()

# Construct the optimizer (Adamax in this case)
optimizer = torch.optim.Adamax(model.parameters(), lr = 0.01)

# Gradient Descent
losses = []
epochs = 3000
epoch = 0
train_error = 0
max_error = ((n_in+2)*(n_h+2)-3)/n.size/2
print(max_error)
while (epoch<epochs or train_error>max_error):
    running_average_loss = 0
    train_data = DataLoader(Xy_train, batch_size=batch_size, shuffle=True, num_workers=0)
    for i, data in enumerate(train_data): # loop through batches
        X_batch, y_batch = data[:,0], data[:,1]
        X_batch, y_batch = X_batch[:,None], y_batch[:,None]

        # Zero gradients
        optimizer.zero_grad()
        # Forward pass: Compute predicted y by passing x to the model
        y_pred = model(X_batch)
        # Compute and print loss
        loss = criterion(y_pred, y_batch)
        # perform a backward pass (backpropagation)
        loss.backward()
        # Update the parameters
        optimizer.step()
        
        running_average_loss += loss.detach().item()
    losses.append(running_average_loss/(i+1))
    if epoch % (epochs/10) == (epochs/10-1):
        print('epoch: ', epoch,' loss: ', losses[epoch])
    train_error = losses[epoch]
    epoch += 1


# visualizing the error after each epoch
cut = int(epoch*0.1)
fig, ax = plt.subplots(1, 2, figsize=(14,3))
fig.suptitle("Error on training data after each epoch")
ax[0].plot(np.arange(0, cut+1), np.array(losses[:cut+1]))
ax[0].set_ylabel('epochs 0 to %i' %cut)
ax[1].plot(np.arange(cut+1, epoch), np.array(losses[cut+1:]))
ax[1].set_ylabel('epochs '+str(cut+1)+' to '+str(epoch-1))
plt.show()

y_predicted = np.squeeze(model(torch.from_numpy(X[:,None])).detach().numpy())
fig, ax = plt.subplots(figsize=(16,9))
plt.plot(180/np.pi*n, 1e-3*y_predicted, label='prediction')
plt.plot(180/np.pi*n, 1e-3*func, '--', label='real')
plt.legend(fontsize=13)
plt.title("Prediction of neural network vs real torque curve over a period",fontsize=14)
ax.set_xlabel("θ (°)",fontsize=14)
ax.set_ylabel("torque (N*m)",fontsize=14)
plt.show()
