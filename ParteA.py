import numpy as np
import matplotlib.pyplot as plt
from control import tf, feedback, step_response, c2d, poles, dcgain


Gp_s = tf([32.4,3240],[1, 0.15, -4.9896])


t = np.linspace(0, 3, 1000) 

print("Polos em S e ganho")
print(f"Polos: {poles(Gp_s)}")
print(f"Ganho: {dcgain(Gp_s)}")

t_continuo, c_continuo = step_response(Gp_s, t)
plt.figure(1)
plt.plot(t_continuo, c_continuo, label='resposta ao degrau em S')


T = .01
Gp_z = c2d(Gp_s,T,method='zoh')


print("Polos em Z e ganho")
print(f"Polos: {poles(Gp_z)}")
print(f"Ganho: {dcgain(Gp_z)}")

t_d = np.arange(0, 3, T)

t_discreto, c_discreto = step_response(Gp_z,t_d)

t_zoh = np.repeat(t_discreto,2)[1:]
c_zoh = np.repeat(c_discreto,2)[:-1]

plt.step(t_zoh,c_zoh,label=' resposta ao degrau em Z' )



T = 0.01  # periodo de amostragem (s)
numz = [0.324, 0]
denz = [1, -1.999, 0.9985]
Gz = tf(numz, denz, T)

print("Polos em Z e ganho(valores joao)")
print(f"Polos: {poles(Gz)}")
print(f"Ganho: {dcgain(Gz)}")

plt.legend()
plt.grid()
plt.show()