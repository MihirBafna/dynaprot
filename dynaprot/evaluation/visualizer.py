import numpy as np
import plotly.graph_objects as go
from scipy.stats import multivariate_normal


def generate_ellipsoid_with_pdf(mean, cov, num_points=50):
    """
    Generate a 3D ellipsoid and its probability density values.

    Parameters:
        mean (np.ndarray): Mean of the Gaussian (shape: [3]).
        cov (np.ndarray): Covariance matrix of the Gaussian (shape: [3, 3]).
        num_points (int): Number of points to generate for the surface.

    Returns:
        tuple: Meshgrid coordinates (X, Y, Z) of the ellipsoid surface and the PDF values.
    """
    # Eigen-decomposition of the covariance matrix
    eigenvalues, eigenvectors = np.linalg.eigh(cov)
    
    # Create a sphere
    u = np.linspace(0, 2 * np.pi, num_points)
    v = np.linspace(0, np.pi, num_points)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))

    # Scale the sphere to the ellipsoid using the eigenvalues
    radii = np.sqrt(eigenvalues)
    ellipsoid = np.array([x * radii[0], y * radii[1], z * radii[2]])

    # Rotate the ellipsoid using the eigenvectors
    ellipsoid_rotated = np.einsum('ij,jkl->ikl', eigenvectors, ellipsoid)

    # Translate the ellipsoid to the mean
    X = ellipsoid_rotated[0] + mean[0]
    Y = ellipsoid_rotated[1] + mean[1]
    Z = ellipsoid_rotated[2] + mean[2]

    # # Compute PDF values for each point
    # points = np.stack([X.ravel(), Y.ravel(), Z.ravel()], axis=1)
    # pdf = multivariate_normal(mean=mean, cov=cov).pdf(points)
    # PDF = pdf.reshape(X.shape)

    return X, Y, Z

def plot_3d_gaussian_ellipsoids(means, covariances, num_points=50, save_path="./ellipsoids.png"):
    """
    Plot multiple 3D Gaussian ellipsoids with gradient based on PDF.

    Parameters:
        means (list of np.ndarray): List of 3D means.
        covariances (list of np.ndarray): List of 3x3 covariance matrices.
        num_points (int): Number of points to generate for the ellipsoid surfaces.
    """
    fig = go.Figure()

    for i, (mean, cov) in enumerate(zip(means, covariances)):
        # Generate the ellipsoid and its PDF
        X, Y, Z = generate_ellipsoid_with_pdf(mean, cov, num_points)

        # Add ellipsoid surface with PDF as gradient
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z,
            # surfacecolor=PDF,  # Use PDF values for color gradient
            colorscale='Viridis',
            opacity=0.8,
            name=f'Gaussian {i+1}',
            showscale=False
        ))

    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        title="3D Gaussian Ellipsoids with PDF Gradient"
    )
    return fig
    fig.write_html(save_path)


def plot_3d_gaussian_comparison(means, ground_truth_cov, predicted_cov, num_points=50, save_path="./ellipsoid_comparison.html"):
    """
    Plot 3D Gaussian ellipsoids for ground truth and predicted covariance matrices.

    Parameters:
        means (np.ndarray): 3D mean vectors.
        ground_truth_cov (np.ndarray): Ground truth 3x3 covariance matrix.
        predicted_cov (np.ndarray): Predicted 3x3 covariance matrix.
        num_points (int): Number of points to generate for the ellipsoid surfaces.
        save_path (str): File path to save the plot as an HTML file.
    """
    fig = go.Figure()

    for i, (mean, cov) in enumerate(zip(means, ground_truth_cov)):
        # Generate the ellipsoid and its PDF
        X, Y, Z = generate_ellipsoid_with_pdf(mean, cov, num_points)

        # Add ellipsoid surface with PDF as gradient
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z,
            # surfacecolor=PDF,  # Use PDF values for color gradient
            colorscale='teal',
            opacity=0.8,
            name=f'Gaussian {i+1}',
            showscale=False
        ))
        
    for i, (mean, cov) in enumerate(zip(means, predicted_cov)):

        X, Y, Z = generate_ellipsoid_with_pdf(mean, cov, num_points)
        X_pred, Y_pred, Z_pred = generate_ellipsoid_with_pdf(mean, cov, num_points)
        fig.add_trace(go.Surface(
                    x=X_pred, y=Y_pred, z=Z_pred,
                    colorscale='Burg',
                    opacity=0.6,
                    name='Predicted',
                    showscale=False
                ))

    fig.update_layout(
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z'
        ),
        margin=dict(l=0, r=0, b=0, t=0),
        title="3D Gaussian Ellipsoids Comparison"
    )
    return fig

    # fig.write_html(save_path)
    print(f"3D Gaussian comparison plot saved to {save_path}")

# plot_3d_gaussian_ellipsoids(means.numpy()[:20], covars.numpy()[:20], num_points=50)
