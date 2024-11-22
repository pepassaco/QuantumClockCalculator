import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from matplotlib import rcParams
import sys
import time
from scipy.linalg import expm
from scipy.integrate import solve_ivp

def create_harmonic_oscillator(d, f):
    """Create the Hamiltonian for quantum harmonic oscillator"""
    diag = np.arange(d)
    return np.diag(2*np.pi/f*(diag + 0.5))

def create_phase_basis(d, k):
    """Create phase basis state for given k"""
    phase_state = np.zeros(d)
    phase_state[k] = 1
    return phase_state

def random_density_matrix(d):
    """Create a random density matrix"""
    # Create random complex matrix
    A = np.random.uniform(size=(d,d)) + 1j * np.random.uniform(size=(d,d))
    # Make it Hermitian
    A = A + A.conj().T
    # Make it positive semidefinite
    A = A @ A.conj().T
    # Normalize
    return A / np.trace(A)

def create_V_operator(d, ko, phaseOp):
    """Create V operator based on flags"""
    if phaseOp:
        # Create projector onto ko-th phase eigenstate
        phase_state = create_phase_basis(d, ko)
        V = np.outer(phase_state, phase_state.conj())
    else:
        # Create random Hermitian positive matrix
        V = random_density_matrix(d)
    return V

def change_basis_matrix(d):
    """Create the change of basis matrix from phase to number basis"""
    # Fourier transform matrix for change of basis
    omega = np.exp(2j * np.pi / d)
    F = np.array([[omega**(j*k) / np.sqrt(d) for k in range(d)] for j in range(d)])
    return F

def create_rho(d, kv, phasePsi):
    """Create density matrix based on flags"""
    if phasePsi:
        # Create pure state in phase basis
        phase_state = create_phase_basis(d, kv)
        rho_phase = np.outer(phase_state, phase_state.conj())
        # Change to number basis
        F = change_basis_matrix(d)
        rho = F @ rho_phase @ F.conj().T
    else:
        rho = random_density_matrix(d)
    return rho

def integrand_Xmu(t, H, V):
    """Integrand for Xmu calculation"""
    # Calculate e^(iHt)
    expAt = expm(1j * H * t - V * t)
    expAdagat = expm(-1j * H * t - V * t)
    
    return expAt @ expAdagat

def integrand_Xsigma(t, H, V):
    """Integrand for Xsigma calculation"""
    return 2 * t * integrand_Xmu(t, H, V)

def calculate_X_operators(H, V, t_max=10, rtol=1e-4, atol=1e-2):
    """Calculate Xmu and Xsigma through numerical integration using solve_ivp"""
    d = H.shape[0]
    
    # Reshape matrices to vectors for ODE solver
    def reshape_to_vector(matrix):
        return matrix.reshape(-1)
    
    def reshape_to_matrix(vector):
        return vector.reshape(d, d)
    
    # Define the system of ODEs
    def ode_system(t, y):

        # Calculate derivatives
        dXmu = integrand_Xmu(t, H, V)
        dXsigma = integrand_Xsigma(t, H, V)
        
        return np.concatenate([reshape_to_vector(dXmu), reshape_to_vector(dXsigma)])
    
    # Initial conditions (zero matrices)
    y0 = np.zeros(2*d*d, dtype=complex)
    
    # Solve the system
    solution = solve_ivp(
        ode_system,
        (0, t_max),
        y0,
        method='RK45',
        rtol=rtol,
        atol=atol,
        dense_output=True
    )
    
    # Extract final values
    final_state = solution.y[:,-1]
    Xmu = reshape_to_matrix(final_state[:d*d])
    Xsigma = reshape_to_matrix(final_state[d*d:])
    
    return Xmu, Xsigma

def compute_expected_values(rho, Xmu, Xsigma):
    """Compute expected values of Xmu and Xsigma with respect to density matrix rho"""
    # Expected value is Tr(rho * X)
    expected_Xmu = np.trace(rho @ Xmu)
    expected_Xsigma = np.trace(rho @ Xsigma)
    return expected_Xmu, expected_Xsigma

def create_plot(x_data, y_data, axvlineHypo=None, logScale = False, title="", xlabel="x", ylabel="y", 
                         err=None, figure_size=(8, 6), save_path=None):
    """
    Create a publication-quality scientific plot.
    
    Parameters:
    -----------
    x_data : array-like
        Data for x-axis
    y_data : array-like
        Data for y-axis
    axvlineHypo : float
        Draws a vertical line here
    title : str, optional
        Plot title
    xlabel : str, optional
        Label for x-axis
    ylabel : str, optional
        Label for y-axis
    err : array-like, optional
        Error bars for y_data
    figure_size : tuple, optional
        Figure size in inches (width, height)
    save_path : str, optional
        Path to save the figure
        
    Returns:
    --------
    fig, ax : tuple
        Matplotlib figure and axis objects
    """
    # Set up the plot style
    plt.style.use('default')  # Reset to default style
    rcParams['font.family'] = 'serif'
    rcParams['font.serif'] = ['Computer Modern Roman']
    rcParams['text.usetex'] = True
    rcParams['axes.labelsize'] = 12
    rcParams['xtick.labelsize'] = 10
    rcParams['ytick.labelsize'] = 10
    rcParams['legend.fontsize'] = 10
    rcParams['figure.figsize'] = figure_size
    rcParams['axes.grid'] = True
    rcParams['grid.alpha'] = 0.3
    rcParams['axes.facecolor'] = '#f0f0f0'

    # Create the plot
    fig, ax = plt.subplots()
    
    if err is not None:
        ax.errorbar(x_data, y_data, yerr=err, fmt='o', color='#1f77b4',
                   capsize=3, capthick=1, elinewidth=1, markersize=5,
                   label='Data')
    else:
        ax.plot(x_data, y_data, 'o-', color='#1f77b4', markersize=5,
                label='Data')
    
    if axvlineHypo is not None:
        ax.axvline(axvlineHypo, linestyle='dashed', color='orange')

    # Customize the plot
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if logScale:
        ax.set_xscale('log')
    # Adjust layout
    plt.tight_layout()
    
    # Save if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    return fig, ax

def progress_bar(current, total, start_time, bar_length=50, prefix='Progress:', suffix='Complete'):
    """
    Creates a progress bar in the terminal that updates in place and shows elapsed time.
    
    Args:
        current (int): Current iteration
        total (int): Total iterations
        start_time (float): Time when the process started (time.time())
        bar_length (int): Length of the progress bar in characters
        prefix (str): Text before the progress bar
        suffix (str): Text after the progress bar
    """
    filled_length = int(round(bar_length * current / float(total)))
    
    # Create the bar with blocks for filled portion and dashes for remaining
    blocks = '█' * filled_length + '░' * (bar_length - filled_length)
    
    # Calculate percentage
    percent = round(100.0 * current / float(total), 1)
    
    # Calculate elapsed time
    elapsed_time = time.time() - start_time
    time_str = f"({elapsed_time:.1f}s)" if current < total else f"(Completed in {elapsed_time:.1f}s)"
    
    # Create the progress bar string
    bar = f'\r{prefix} |{blocks}| {percent}% {suffix} {time_str}'
    
    # Write to terminal and flush
    sys.stdout.write(bar)
    sys.stdout.flush()
    
    # Print a newline when complete
    if current == total:
        print()

def save_array(array, filename, directory='.'):
    """
    Save a numpy array to a file.
    
    Parameters:
    -----------
    array : numpy.ndarray
        The array to save
    filename : str
        Name of the file (without .npy extension)
    directory : str
        Directory where to save the file (default: current directory)
    """
    import os
    
    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)
    
    # Create full path
    full_path = os.path.join(directory, f"{filename}.npy")
    
    # Save the array
    np.save(full_path, array)
    print(f"Array saved to: {full_path}")

def main():
    # System parameters
    d = 10  # Dimension of Hilbert space
    f = 10 # Frecuencia de H
    ko = d-1  # Index for phase operator
    kv = 0  # Index for initial density matrix
    phaseOp = True  # Flag for V operator
    phasePsi = True  # Flag for density matrix

    nNorms = 3
    normMin = 1e-5
    normMax = 2
    logSep = True

    t_max = int(1e4)

    name = "phaseEigenstate_" if phaseOp else "random_"
    
    # Create Hamiltonian
    H = create_harmonic_oscillator(d,f)
    
    # Create V operator
    V = create_V_operator(d, ko, phaseOp)

    '''
    if not phaseOp:
        print("Operador V aleatorio:")
        print(V)
    '''
    
    # Change V to number basis
    F = change_basis_matrix(d)
    Vbase = F.conj().T @ V @ F
    
    # Create density matrix
    rho = create_rho(d, kv, phasePsi)

    

    normsV = np.geomspace(normMin, normMax, nNorms) if logSep else np.linspace(normMin, normMax, nNorms)

    EXmu_norms = []
    EXsigma_norms = []
    quotSigmaMu2 = []

    start_time = time.time()

    for index, norm in enumerate(normsV):
        progress_bar(index, len(normsV), start_time)

        # Calculate X operators
        Xmu, Xsigma = calculate_X_operators(H, norm*Vbase, t_max=t_max)
        
        # Calculate expected values
        exp_Xmu, exp_Xsigma = np.real(compute_expected_values(rho, Xmu, Xsigma))

        EXmu_norms.append(exp_Xmu)
        EXsigma_norms.append(exp_Xsigma)
        quotSigmaMu2.append(exp_Xsigma/(exp_Xmu)**2)

    progress_bar(len(normsV), len(normsV), start_time)
    print("Normas:", normsV)
    print("Mus:", EXmu_norms)
    print("Sigmas:", EXsigma_norms)

    save_array(normsV, name+"norms", directory='./data/')
    save_array(EXmu_norms, name+"mus", directory='./data/')
    save_array(EXsigma_norms, name+"sigmas", directory='./data/')

    axvlineHypo = None

    create_plot(normsV, EXmu_norms, axvlineHypo=axvlineHypo, logScale=logSep, title=r"$\mu$ vs V", xlabel=r"V", ylabel=r"$\mu$", 
                         err=None, figure_size=(8, 6), save_path='./plots/'+name+'mu.pdf')
    create_plot(normsV, EXsigma_norms, axvlineHypo=axvlineHypo, logScale=logSep, title=r"$\sigma$ vs V", xlabel=r"V", ylabel=r"$\sigma$", 
                         err=None, figure_size=(8, 6), save_path='./plots/'+name+'sigma.pdf')
    create_plot(normsV, quotSigmaMu2, axvlineHypo=axvlineHypo, logScale=logSep, title=r"$\sigma/\mu^2$ vs V", xlabel=r"V", ylabel=r"$\sigma/\mu^2$", 
                         err=None, figure_size=(8, 6), save_path='./plots/'+name+'quot.pdf')
    
    return H, V, Vbase, rho, Xmu, Xsigma, exp_Xmu, exp_Xsigma

if __name__ == "__main__":
    H, V, Vbase, rho, Xmu, Xsigma, exp_Xmu, exp_Xsigma = main()