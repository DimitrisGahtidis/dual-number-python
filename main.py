import numpy as np
import matplotlib.pyplot as plt
import time
import dunpy.backward as dn
from dunpy.backward import eps

criterion = lambda x,y: dn.power((x-y),2).mean()

d_feature = 100
dataset = np.random.uniform(-3, 3, (1000, d_feature))

coeff = np.random.uniform(size=(4)) + eps
polynomial = lambda x,c: c[0] + c[1]*x + c[2]*x**2 + c[3]*x**3

accum_steps = 16
log_iter = 10

test = np.linspace(-3, 3, d_feature)
n_epochs = 10
total_loss = 0.
gamma = (5e-1)**(1/(n_epochs))
lr = 5e-3
for epoch in range(n_epochs):
    if epoch != 0:
        lr *= gamma
    then = time.time()
    np.random.shuffle(dataset)
    for step, x in enumerate(dataset):
        x = np.atleast_2d(x)
        output = polynomial(x, coeff)
        target = np.sin(x)
        loss = criterion(output, target)/(accum_steps)
        loss.grad()
        total_loss += loss.re/(log_iter)
        if (step+1) % accum_steps == 0 or step == len(dataset):
            if ((step+1)//accum_steps) % max(((len(dataset)//accum_steps)//(log_iter)),1) == 0:
                print(f"batch {(step+1)//accum_steps}/{len(dataset)//accum_steps} | loss = {np.round(total_loss,4)} | lr = {np.round(lr*1000,3)} e-3")
            total_loss = 0.
            dn.update(coeff, lr)
            dn.zero_grad(coeff)
    if (epoch+1) > 1:
        plt.cla()
        plt.title(f"Epoch {epoch+1}")
        plt.plot(test, np.sin(test), color="black", label="sin curve")
        plt.plot(test, np.vectorize(lambda z: z.re)(polynomial(test, coeff)), color="black", linestyle="--", label="cubic fit")
        plt.xlim(-np.pi*1.1, np.pi*1.1)
        plt.ylim(-1*1.1,1*1.1)
        plt.legend()
        dn.update(coeff, lr)
        dn.zero_grad(coeff)
        plt.pause(0.1)
    now = time.time()
    print(f"|| End of epoch {epoch+1} | epoch time = {np.round(now-then,3)} ||")