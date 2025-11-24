import numpy as np
import matplotlib.pyplot as plt
import control as ct
import warnings

# Ignorar avisos de depreciação da biblioteca control para limpar o terminal
warnings.filterwarnings("ignore")

# ==============================================================================
# 1. CARREGAMENTO DOS DADOS
# ==============================================================================
try:
    dados = np.load('pid_dados.npz')
    Kp = float(dados['Kp'])
    Ki = float(dados['Ki'])
    Kd = float(dados['Kd'])
    T = float(dados['T'])
    print(f"--- DADOS CARREGADOS ---")
    print(f"Ganhos: Kp={Kp:.4f}, Ki={Ki:.4f}, Kd={Kd:.4f}")
    print(f"Tempo de Amostragem T={T}s")
except FileNotFoundError:
    print("ERRO: Execute 'ParteB.py' primeiro para gerar os ganhos.")
    exit()

# ==============================================================================
# 2. DEFINIÇÃO DO SISTEMA
# ==============================================================================
# Planta G(z)
num_g = [0.324, 0]
den_g = [1, -1.999, 0.9985]
Gz = ct.tf(num_g, den_g, T)

# Controlador PID C(z)
q2 = Kp + Ki + Kd
q1 = -(Kp + 2*Kd)
q0 = Kd
num_pid = [q2, q1, q0]
den_pid = [1, -1, 0]
Cz = ct.tf(num_pid, den_pid, T)

# Malha Aberta e Fechada
Lz = Cz * Gz
sys_cl = ct.feedback(Lz, 1)

# Função de Transferência do Esforço de Controle U(z)
sys_u = ct.feedback(Cz, Gz)

# ==============================================================================
# 3. ANÁLISE GRÁFICA
# ==============================================================================

# --- GRÁFICO 1: LUGAR DAS RAÍZES (CORRIGIDO) ---
plt.figure(figsize=(8, 6))

# Chamada simplificada para evitar o erro de unpack (rlist, klist)
ct.root_locus(Lz, plot=True, grid=True)

# Plotar manualmente os polos de malha fechada (usando ct.poles conforme pedido)
polos_mf = ct.poles(sys_cl)
plt.plot(np.real(polos_mf), np.imag(polos_mf), 'rX', markersize=12, markeredgewidth=3, label='Polos Finais')

# Ajuste Fino da Escala
plt.xlim([-1.2, 1.2])
plt.ylim([-1.2, 1.2])
plt.title("Lugar das Raízes (Zoom na Estabilidade)")
plt.legend(loc='upper right')


# --- GRÁFICO 2: RESPOSTA AO DEGRAU ---
plt.figure(figsize=(10, 5))
t, y = ct.step_response(sys_cl, T=4.0)
plt.step(t, y, where='post', linewidth=2, label='Saída y[k]')
plt.axhline(1.0, color='r', linestyle='--', label='Referência (1.0)')
plt.axhline(1.3, color='orange', linestyle=':', label='Limite 30%')

# Métricas
info = ct.step_info(sys_cl)
Mp = info['Overshoot']
ts = info['SettlingTime']
print(f"\n--- DESEMPENHO ---")
print(f"Sobressinal (Mp): {Mp:.2f}% (Requisito: <= 30%)")
print(f"Tempo de Acomodação (ts): {ts:.4f}s")

plt.title(f"Resposta ao Degrau\nMp={Mp:.1f}%, ts={ts:.2f}s")
plt.xlabel("Tempo (s)")
plt.ylabel("Amplitude")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()


# --- GRÁFICO 3: SINAL DE CONTROLE ---
plt.figure(figsize=(10, 5))
t_u, u = ct.step_response(sys_u, T=4.0)
plt.step(t_u, u, where='post', color='green', linewidth=1.5, label='Controle u[k]')

u_max = np.max(np.abs(u))
print(f"\n--- ESFORÇO DE CONTROLE ---")
print(f"Pico Máximo de u[k]: {u_max:.2f}")

plt.title(f"Sinal de Controle u[k]\nPico Máximo: {u_max:.2f}")
plt.xlabel("Tempo (s)")
plt.ylabel("Amplitude")
plt.grid(True, linestyle='--', alpha=0.6)
plt.legend()

plt.show()