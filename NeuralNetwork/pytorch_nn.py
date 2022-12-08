import torch, pandas, os, numpy

import torch.utils.data as torch_data

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
HIDDEN_NETWORK_SIZE = [5, 10, 25, 50, 100]
# HIDDEN_NETWORK_SIZE = [5]
DEPTHS = [3,5,9]
T = 10
MODES = [0,1]
BATCH_SIZE = 10

bankn_train = "../data/bank-note/train.csv" if os.path.isfile("../data/bank-note/train.csv") else "data/bank-note/train.csv"
bankn_test = "../data/bank-note/test.csv" if os.path.isfile("../data/bank-note/test.csv") else "data/bank-note/test.csv"

def xavier(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.xavier_normal_(m.weight)
        m.bias.data.fill_(0.01)
##

def he(m):
    if isinstance(m, torch.nn.Linear):
        torch.nn.init.kaiming_normal_(m.weight)
        m.bias.data.fill_(0.01)
##

class NeuralNetworkPT(torch.nn.Module):
    def __init__(self, depth, width, mode=0) -> None:
        super(NeuralNetworkPT, self).__init__()

        if mode == 0:
            # tanh / He
            self.activation = torch.nn.Tanh()
            self.trainingMethod = he
        elif mode == 1:
            # ReLU / Xavier
            self.activation = torch.nn.ReLU()
            self.trainingMethod = xavier
        else:
            raise Exception("Nope!")

        self.input_layer = torch.nn.Sequential(torch.nn.Linear(4, width), self.activation)
        self.layers = torch.nn.ModuleList([])
        
        for _ in range(depth-2):
            self.layers.append(torch.nn.Sequential(torch.nn.Linear(width, width), self.activation))
        
        self.output_layer = torch.nn.Linear(width, 1)
    ##

    def forward(self, x):
        x = self.input_layer(x)

        for layer in self.layers:
            x = layer(x)

        return self.output_layer(x)
    ##
###

class Dataset(torch_data.Dataset):
    def __init__(self, file) -> None:
        training = pandas.read_csv(file, header=None)

        row_count = training.values.shape[0]
        col_count = training.values.shape[1]

        x = []
        y = []

        for i in range(row_count):
            dp = training.iloc[i].tolist()
            
            dp = list(map(lambda x: numpy.float32(x), dp))

            x.append(dp[0:col_count-1])
            y.append(dp[col_count-1])
        ##

        self.x = numpy.array(x)
        self.y = numpy.array(y)

        self.length = len(self.x)
    ##

    def __getitem__(self, index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.length
    ##
###

def train_nn(loader, model, loss, optimizer):
    model.train()
    loss_list = []

    for batch, (x,y) in enumerate(loader):
        x = x.to(DEVICE)
        y = y.to(DEVICE)

        prediction = model(x)
        l = loss(torch.reshape(prediction, y.shape), y)

        # backprop
        optimizer.zero_grad()
        l.backward()
        optimizer.step()

        if batch % BATCH_SIZE == 0:
            loss_list.append(l.item())

    return loss_list

def test_nn(loader, model, loss):
    tot_loss = 0
    nbatch = len(loader)
    model.eval()

    with torch.no_grad():
        for x, y in loader:
            x = x.to(DEVICE)
            y = y.to(DEVICE)

            prediction = model(x)

            tot_loss += loss(torch.reshape(prediction, y.shape), y).item()

    return tot_loss / nbatch

train_set = Dataset(bankn_train)
train_loader = torch_data.DataLoader(train_set, batch_size=BATCH_SIZE)

test_set = Dataset(bankn_test)
test_loader = torch_data.DataLoader(test_set, batch_size=BATCH_SIZE)

if __name__ == "__main__":
    combos = len(HIDDEN_NETWORK_SIZE) * len(DEPTHS) * len(MODES)

    print("Beginning training...")
    results = {}
    for s in HIDDEN_NETWORK_SIZE:
        for d in DEPTHS:
            for m in MODES:
                key = (s,d,m)
                model = NeuralNetworkPT(d, s, mode=m).to(DEVICE)
                model.apply(model.trainingMethod)

                loss = torch.nn.MSELoss()
                optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

                train_loss = []
                for t in range(T):
                    epoch_loss = train_nn(train_loader, model, loss, optimizer)
                    train_loss.append(epoch_loss)
                
                test_loss = test_nn(test_loader, model, loss)

                results[key] = [test_loss, train_loss]
            ##
        ##
    ##

    print(f"Finished {combos} different combinations!")
    
    print("Total Epochs:", T)
    for r in results:
        res = results[r]
        result_str = f"Training Error: {numpy.mean(res[1][-1]):>8f}; Test Error: {str(res[0])}"
        print(f"Network of with width {r[0]}, depth {r[1]}, and using {'ReLU' if r[2] == 1 else 'Tanh'}:\n\t{result_str}")
    ##
##
