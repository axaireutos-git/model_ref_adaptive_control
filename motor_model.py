import numpy as np
from scipy.integrate import solve_ivp
import matplotlib.pyplot as plt
import torch


Am = np.array([[ 0,    1],
               [-24, -10]])

Bm = np.array([0, 24])

P = 1/96*np.array([[140, 2],
                   [ 2,  5]])

Γx = np.array([[1, 0],
               [0, 2]])
Γr = 1
Γθ = 10

N = 50
J, Km, b = 4.5*np.float_power(10,-5), 0.19, 8*np.float_power(10,-4)
A = np.array([[0,   1 ],
              [0, -b/J]])

B = np.array([0, 1])

np.random.seed(6)
def uncertainty(x, u, limit=0):
    if (limit!=0): return (1+np.sign(limit)*u)*x
    else:          return np.random.uniform((1-u)*x,(1+u)*x)

p = 0.05
J_est, Km_est, b_est = uncertainty(J,p), uncertainty(Km,p), uncertainty(b,p)

I = 1
Λ = Km/J*I
λ = Km_est/J_est*I
print(λ/Λ)

Kx_ideal = 1/Λ*np.array([-24, -10+b/J])
Kx0 = 1/λ*np.array([-24, -10 + b_est/J_est])
Kr_ideal = np.float64(24/Λ)
Kr0 = 24/λ

def external_torque(x):
    return 1e-3*np.float_power(np.cos(2*x),2)*np.sin(3*x)


def real_vs_adaptive_system(t, Z, degrees, Dt=1, 
                            ext_torque=False, nn_torque_model=None, continuous=False):
    x = Z[0:2]
    xm = Z[2:4]
    Kx = np.array([Z[4], Z[5]])
    Kr = Z[6]
    Θ = Z[7]
    
    ref = np.floor(t/Dt)
    if (continuous): ref = t/Dt
    r = np.pi*degrees/180*ref

    TL, Φ = 0, 0
    if (ext_torque):
        TL = external_torque(x[0])              # TL = external unknown torque
        if (nn_torque_model!=None):
            x_torch = torch.from_numpy(np.reshape(x[0],(1,1)))
            Φ = np.squeeze(nn_torque_model(x_torch).detach().numpy())
    
    dXm = np.dot(Am,xm) + np.dot(Bm,r)
    
    u = np.dot(Kx.T,x) + np.dot(Kr.T,r) + np.dot(Θ.T, Φ)
    
    dX = np.dot(A,x) + np.dot(np.dot(B,Λ),u)
    dX[1] -= TL/J
    
    e = x - xm
    M = -np.dot(np.dot(e.T,P),B)
    dKx = np.dot(np.dot(Γx,x),M)
    dKr = np.dot(np.dot(Γr,r),M)
    dΘ = np.dot(np.dot(Γθ,Φ),M)
    
    DZ = np.concatenate((dX, dXm, dKx, [dKr, dΘ]))
    return DZ


def simulate_system(t_span, degrees=5, Dt=3, 
                    ext_torque=False, nn_torque_model=None, continuous=False):
    return solve_ivp(fun=real_vs_adaptive_system, t_span=[0,t_span], 
                     y0=[0, 0, 0, 0, Kx0[0], Kx0[1], Kr0, 0], 
                     dense_output=True, method='RK45', 
                     args=(degrees,Dt,ext_torque,nn_torque_model, continuous))

def plot_response(solution, x1, string=""):
    fig, ax = plt.subplots(figsize=(16,9))
    plt.plot(solution.t, 180/np.pi*solution.y[x1,:], label='System')
    plt.plot(solution.t, 180/np.pi*solution.y[x1+2,:], '--', label='Reference model')
    plt.legend(fontsize=13)
    ax.set_xlabel("time (sec)",fontsize=14)
    if (x1==0):
        plt.title("Stepper motor response"+string+
                  ":\nplot of rotation angle over time",fontsize=14)
        ax.set_ylabel("θ (°)",fontsize=14)
    elif (x1==1):
        plt.title("Stepper motor response"+string+
                  ":\nplot of angular velocity over time",fontsize=14)
        ax.set_ylabel("ω (rad/sec)",fontsize=14)
    plt.show()
    return

def plot_parameters(solution,nonlinear=False):
    fig, ax = plt.subplots(3+int(nonlinear), 1, sharex=True, figsize=(15,11))
    ideal_K = np.repeat(np.array([Kx_ideal[0], Kx_ideal[1], Kr_ideal])[:,None], 
                        np.size(solution.t),axis=1)
    for i in range(3):
        ax[i].plot(solution.t, solution.y[4+i,:], label='Estimated')
        ax[i].plot(solution.t, ideal_K[i], '--', label='Ideal')
        ax[i].legend(fontsize=13)
    if (nonlinear): ax[3].plot(solution.t, solution.y[7,:])
    fig.suptitle("Evolution of adaptive parameters over time",fontsize=14,y=0.91)
    ax[0].set_ylabel("Kx1",fontsize=14)
    ax[1].set_ylabel("Kx2",fontsize=14)
    ax[2].set_ylabel("Kr",fontsize=14)
    if (nonlinear): ax[3].set_ylabel("Θ",fontsize=14)
    ax[2+int(nonlinear)].set_xlabel("time (sec)",fontsize=14)
    plt.show()
    return


solution = simulate_system(t_span=28)
plot_response(solution,0)
plot_response(solution,1)
plot_parameters(solution)


t = solution.t
x = solution.y
step = np.floor(t)
r = np.pi*5/180*(np.divmod(step,3)[0])
u = x[4]*x[0]+x[5]*x[1] + x[6]*r
ia = -I*u*np.sin(50*x[0])
ib = I*u*np.cos(50*x[0])

fig, ax = plt.subplots(figsize=(16,9))
plt.plot(t, 1000*ia, label='ia')
plt.plot(t, 1000*ib, label='ib')
plt.legend(fontsize=13)
ax.set_xlabel("time (sec)",fontsize=14)
ax.set_ylabel("I (mA)",fontsize=14)
fig.suptitle("Phase currents of stepper motor",fontsize=14,y=0.91)
plt.show()


#   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #   #

solution = simulate_system(t_span=28,ext_torque=True,nn_torque_model=model)
plot_response(solution, 0, string=
              " with load\nand neural network as estimator of unknown torque")

plot_parameters(solution,nonlinear=True)

solution = simulate_system(t_span=28,ext_torque=True)
plot_response(solution, 0, string=
              " with load,\nbut without estimation of unknown torque")