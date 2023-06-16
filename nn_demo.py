###############################       Wide Deep          ########################### 

 

import torch 

import torchvision 

 

# mock data 

import random   

 

XW,XD,LABELS = [],[],[] 

 

for i in range(10): 

    XW.append(torch.randn(50)) 

    XD.append(torch.randn(3,128,128)) 

    LABELS.append(random.randint(0,1)) 

 

# X_w[0],X_d[0],labels[0] 

 

class MyDataset(torch.utils.data.Dataset): 

    def __init__(self,X_w,X_d,labels): 

        super().__init__() 

        self.X_w = X_w 

        self.X_d = X_d 

        self.labels = labels 

         

    def __len__(self): 

        return len(self.labels) 

     

    def __getitem__(self,index): 

        return self.X_w[index], self.X_d[index], self.labels[index] 

 

class Net(torch.nn.Module): 

    def __init__(self): 

        super(Net,self).__init__()                        

         

        # deep net 

        self.deep1 = torch.nn.Conv2d(in_channels=3, out_channels=3, kernel_size=(3,3), stride=2, padding=2) 

        self.deep2 = torch.nn.Linear(1*3*32*32, 1500)  

        self.deep3 = torch.nn.Linear(1500, 700)  

        self.deep4 = torch.nn.Linear(700, 100)  

         

        # wide net       

        self.wide5 = torch.nn.Linear(100+50,10) 

        self.wide6 = torch.nn.Linear(10,2) 

     

    def compile(self):         

        self.loss = torch.nn.CrossEntropyLoss() 

        self.optimizer = torch.optim.Adam(self.parameters(),lr=0.0001) 

        self.activation = torch.nn.functional.relu 

        self.pool = torch.nn.MaxPool2d(2,2) 

        self.epoch = 50 

        self.gpu = torch.cuda.is_available() 

         

    def forward(self,X_w,X_d):         

 

        #deep 

        x = X_d 

        x=self.pool(self.activation(self.deep1(x))).view(-1,3*32*32) #这里要用view而不用flatten，因为一个batch都超过1条的数据，会形成batch size的tensor，flatten会把所有batch拉成一行 

        x = self.activation(self.deep2(x)) 

        x = self.activation(self.deep3(x)) 

        x = self.activation(self.deep4(x)) 

         

        #wide 

        x = torch.cat([x,X_w],1) # 二维tensor拼接 

        x = self.activation(self.wide5(x)) 

        x = self.activation(self.wide6(x)) 

         

        return x 

     

    def fit(self, ds, epoch_size, batch_size): 

        # train init 

        print("Training start...") 

        data_loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True) 

        net = self.train() 

        if self.gpu: 

            net = net.cuda() 

         

        for epoch in range(epoch_size):             

            # epoch error 

            ep_cor = 0.0 

            ep_size = 0.0 

                         

            # epoch train 

            for X_w,X_d,label in data_loader: 

                if self.gpu: 

                    X_w,X_d,label = X_w.cuda(),X_d.cuda(),label.cuda() 

                self.optimizer.zero_grad() 

                y = net(X_w,X_d) 

                loss = self.loss(y,label) 

                loss.backward() 

                self.optimizer.step()                                

             

                # epoch error 

                y_label = torch.max(y,1)[1] 

                ep_cor += (y_label==label).sum().item() 

                ep_size += label.size(0) 

                             

            # epoch error 

            print("epoch %d, accuracy=%.3f" % (epoch+1, ep_cor/ep_size)) 

        print("Training end") 

        return 0 

     

    def predict(self,ds,batch_size=1000): 

        data_loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False) 

        net=self.eval() 

        with torch.no_grad(): 

            for X_w,X_d,_ in data_loader: 

                if self.gpu: 

                    X_w,X_d = X_w.cuda(),X_d.cuda() 

                y = net(X_w,X_d)      

        return torch.max(y,1) 

 

         

 

ds = MyDataset(XW,XD,LABELS) 

 

 

net=Net() 

net.compile() 

net.fit(ds,100,3) 

 

ds2 = MyDataset(XW,XD,XD) 

net.predict(ds2) 

 

 
