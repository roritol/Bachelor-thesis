import numpy as np
from scipy import stats

import matplotlib as mpl
import matplotlib.pyplot as plt


def plot_line(ax, v, **kwargs):
    ax.axline((0, 0), v, **kwargs)


def main():

    n_samples = 1000

    X = np.array([
        [+.5, 1],
        [-.5, 0],
        [+.5, -1],
        [-.5, -2]])

    D = np.array([
        [1, .05],
        [1, .05],
        [1, .05],
        [1, .05]])

    radii = 2 * np.sqrt(2 * D)

    rng = np.random.default_rng(42)

    # [1000 x 2 x 4]
    samples = np.dstack([rng.multivariate_normal(X[i], np.diag(D[i]),
                                                 size=n_samples)
                         for i in range(len(X))])



    # pca on means
    S = np.dot(X.T, X)
    _, vecs_means = np.linalg.eigh(S)

    # pca on distributions
    S += np.diag(np.sum(D, axis=0))
    _, vecs_distrib = np.linalg.eigh(S)


    for jj, k in enumerate(np.arange(1, n_samples, step=5)):
        fig = plt.figure()
        ax = plt.gca()

        # ellipses
        for i in range(len(X)):
            ell = mpl.patches.Ellipse(
                X[i], radii[i, 0], radii[i, 1], 0
            )
            ax.add_artist(ell)
            ell.set_clip_box(ax.bbox)
            ell.set_alpha(0.5)

            plt.scatter(samples[:k, 0, i], samples[:k, 1, i], marker='.',
                        color='gray')

        samples_flat = samples[:k, :, :].reshape(-1, 2)
        S = samples_flat.T @ samples_flat
        _, vecs_samples = np.linalg.eigh(S)

        plot_line(ax, vecs_means[:, 1], label="PCA on means", color='C0', lw=2)
        plot_line(ax, vecs_distrib[:, 1], label="PCA on distributions",
                  color='C1', lw=2)
        plot_line(ax, vecs_samples[:, 1], label="PCA on samples", ls=":",
                  color='k')

        ax.set_xlim(-5, 5)
        ax.set_ylim(-5, 5)
        plt.legend()
        plt.savefig(f"pca{jj:04d}.png")
        plt.close(fig)
        # plt.show()
    # ax.scatter(samples[:, 0], samples[:, 1], marker='.')


if __name__ == '__main__':
    main()

