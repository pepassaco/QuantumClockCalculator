import numpy as np
from scipy.linalg import expm
import matplotlib.pyplot as plt
from matplotlib import rcParams
import sys
import time
import os
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

def create_plot(x_data, y_data, axvlineHypo=None, logScaleX = False, logScaleY = False, title="", xlabel="x", ylabel="y", 
                         err=None, figure_size=(8, 6), save_path=None):
    """
    Create a publication-quality scientific plot.
    
    Parameters:
    -----------
    x_data : array-like
        Data for x-axis
    y_data : array-like
        Data for y-axis
    axvlineHypo : float, optional
        Draws a vertical line here
    logScaleX : boolean, optional
        Log scale for axis X
    logScaleY : boolean, optional
        Log scale for axis Y
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
    rcParams['text.latex.preamble'] = r"\usepackage{amsmath} \usepackage{amssymb}"

    # Create the plot
    fig, ax = plt.subplots()
    
    if err is not None:
        ax.errorbar(x_data, y_data, yerr=err, fmt='o', color='#1f77b4',
                   capsize=3, capthick=1, elinewidth=1, markersize=5,
                   label='Data')
    else:
        ax.plot(x_data, y_data, '-', color='#1f77b4', markersize=5,
                label='Data')
    
    if axvlineHypo is not None:
        ax.axvline(axvlineHypo, linestyle='dashed', color='orange')

    # Customize the plot
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    if logScaleX:
        ax.set_xscale('log')
    if logScaleY:
        ax.set_yscale('log')
    # Adjust layout
    plt.tight_layout()
    
    # Save if path is provided
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        #print(f"Plot saved to: {save_path}")

    plt.clf()
    plt.close('all')

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

def load_array(filename, directory='./data/'):
    """
    Load a numpy array from a file.
    
    Parameters:
    -----------
    filename : str
        Name of the file (without .npy extension)
    directory : str
        Directory where to save the file (default: current directory)
    """

    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)
    
    # Create full path
    full_path = os.path.join(directory, f"{filename}")
    print("Looking for file:", full_path)
    # Load array if exists, otherwise return empty array
    return np.load(full_path) if os.path.exists(full_path) else np.array([])

def save_array(array, filename, directory='./data/'):
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

    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)
    
    # Create full path
    full_path = os.path.join(directory, f"{filename}.npy")

    # Save the array
    np.save(full_path, array)
    #print(f"Array saved to: {full_path}")

def load_data(filenames, directory='./data/'):
    norms = load_array(filenames[0]+".npy", directory).tolist()
    mus = load_array(filenames[1]+".npy", directory).tolist()
    sigmas = load_array(filenames[2]+".npy", directory).tolist()
    quotients = load_array(filenames[3]+".npy", directory).tolist()

    return [norms, mus, sigmas, quotients]

def update_array(new_entry, filename, directory='.', axis=0):
    """
    Updates a NumPy array stored in a .npy file by adding a new entry and saving it.
    
    Parameters:
    filename (str): Path to the .npy file
    new_entry (array-like): New data to append to the existing array
    axis (int): Axis along which to append the new entry (default=0)
    
    Returns:
    numpy.ndarray: The updated array
    """

    # Ensure the directory exists
    os.makedirs(directory, exist_ok=True)

    # Create full path
    full_path = os.path.join(directory, f"{filename}.npy")
    
    try:
        # Load existing array
        existing_array = np.load(full_path)
        
        # Convert new entry to numpy array if it isn't already
        new_entry = np.array(new_entry)
        
        # Make sure new_entry has compatible shape
        if axis == 0:
            if existing_array.shape[1:] != new_entry.shape[1:]:
                raise ValueError("Shape mismatch: new entry is not compatible with existing array")
        else:
            if existing_array.shape[:axis] != new_entry.shape[:axis] or \
               existing_array.shape[axis+1:] != new_entry.shape[axis+1:]:
                raise ValueError("Shape mismatch: new entry is not compatible with existing array")
        
        # Append new entry to existing array
        updated_array = np.append(existing_array, new_entry, axis=axis)
        
        # Save updated array
        np.save(full_path, updated_array)
        
        return updated_array
        
    except FileNotFoundError:
        # If file doesn't exist, create new array from new_entry
        new_array = np.array(new_entry)
        np.save(full_path, new_array)
        return new_array
    
def main():
    # System parameters
    d = 10  # Dimension of Hilbert space
    f = 10 # Frecuencia de H
    ko = d-1  # Index for V's phase operator eigenstate
    kv = 0  # Index for initial density matrix phase operator eigenstate
    phaseOp = True  # Flag for turning the V operator into a projector of the ko-th phase state. Uniformly random positive semidefinite operator if False
    phasePsi = True  # Flag for turning the density matrix of the initial state into the kv-th phase state. Uniformly random density matrix if False


    '''
    # Paramters for studying large ||V||
    nNorms = int(1e3) # Number of different norms to study
    normMin = 1
    normMax = 1e5
    logSep = True # True for logarithmic spacing between norms, False for linear spacing
    rtol=1e-7
    atol=1e-10

    t_max = int(1e8) # Maximum time for numerical integration limit
    '''

    '''
    # Paramters for studying small ||V||
    nNorms = int(1e3) # Number of different norms to study
    normMin = 1e-5
    normMax = 1
    logSep = True # True for logarithmic spacing between norms, False for linear spacing
    rtol=1e-3
    atol=1e-4

    t_max = int(2e9) # Maximum time for numerical integration limit
    '''

    

    axvlineHypo = None

    sufix = "f_"+str(f)+"_d_"+str(d)+"_"
    name = "phaseEigenstate_" if phaseOp else "random_"
    logString = "logScale_" if  logSep else "linScale_"
    endings = ["norms", "mus", "sigmas", "quotients"]
    filenames = [name+sufix+logString+ending for ending in endings]

    directory_data = "./data/"
    directory_plots = "./plots/"



    # Create Hamiltonian
    H = create_harmonic_oscillator(d,f)
    
    # Create V operator
    V = create_V_operator(d, ko, phaseOp)

    '''
    if not phaseOp:
        print("Random V operator:")
        print(V)
    '''
    
    # Change V to number basis
    F = change_basis_matrix(d)
    Vbase = F.conj().T @ V @ F
    
    # Create density matrix
    rho = create_rho(d, kv, phasePsi)

    

    normsV = np.geomspace(normMin, normMax, nNorms) if logSep else np.linspace(normMin, normMax, nNorms)

    [computedNorms, EXmu_norms, EXsigma_norms, quotSigmaMu2] = load_data(filenames, directory_data)
    start_time = time.time()

    for index, norm in enumerate(normsV):
        progress_bar(index, len(normsV), start_time)

        if norm not in computedNorms:
            # Calculate X operators
            Xmu, Xsigma = calculate_X_operators(H, norm*Vbase, t_max=t_max, rtol=rtol, atol=atol)
            
            # Calculate expected values
            exp_Xmu, exp_Xsigma = np.real(compute_expected_values(rho, Xmu, Xsigma))

            # TODO: Fix these if statements
            if computedNorms is not None and norm < computedNorms[index]:
                computedNorms.insert(index,norm)
                EXmu_norms.insert(index,exp_Xmu)
                EXsigma_norms.insert(index,exp_Xsigma)
                quotSigmaMu2.insert(index,exp_Xsigma/(exp_Xmu)**2)
            elif computedNorms is not None and norm > computedNorms[-1-index]:
                computedNorms.append(norm)
                EXmu_norms.append(exp_Xmu)
                EXsigma_norms.append(exp_Xsigma)
                quotSigmaMu2.append(exp_Xsigma/(exp_Xmu)**2)

            save_array(computedNorms, filenames[0], directory=directory_data)
            save_array(EXmu_norms, filenames[1], directory=directory_data)
            save_array(EXsigma_norms, filenames[2], directory=directory_data)
            save_array(quotSigmaMu2, filenames[3], directory=directory_data)

            create_plot(computedNorms, EXmu_norms, axvlineHypo=axvlineHypo, logScaleX=logSep, logScaleY=True, title=r"$\mu$ vs $\lvert\lvert V\rvert\rvert$", xlabel=r"$\lvert\lvert V\rvert\rvert$", ylabel=r"$\mu$", 
                                err=None, figure_size=(8, 6), save_path=directory_plots+filenames[1]+'.pdf')
            create_plot(computedNorms, EXsigma_norms, axvlineHypo=axvlineHypo, logScaleX=logSep, logScaleY=True, title=r"$\mathbb{E}[t^2]$ vs $\lvert\lvert V\rvert\rvert$", xlabel=r"$\lvert\lvert V\rvert\rvert$", ylabel=r"$\mathbb{E}[t^2]$", 
                                err=None, figure_size=(8, 6), save_path=directory_plots+filenames[2]+'.pdf')
            create_plot(computedNorms, quotSigmaMu2, axvlineHypo=axvlineHypo, logScaleX=logSep, logScaleY=False, title=r"$\mathbb{E}[t^2] / \mu^2$ vs $\lvert\lvert V\rvert\rvert$", xlabel=r"$\lvert\lvert V\rvert\rvert$", ylabel=r"$\mathbb{E}[t^2] / \mu^2$", 
                                err=None, figure_size=(8, 6), save_path=directory_plots+filenames[3]+'.pdf')


    progress_bar(len(normsV), len(normsV), start_time)
    #print("Normas:", normsV)
    #print("Mus:", EXmu_norms)
    #print("Sigmas:", EXsigma_norms)
    print("Done! :)")

    

    

    
    
    return H, V, Vbase, rho, Xmu, Xsigma, exp_Xmu, exp_Xsigma

if __name__ == "__main__":
    H, V, Vbase, rho, Xmu, Xsigma, exp_Xmu, exp_Xsigma = main()