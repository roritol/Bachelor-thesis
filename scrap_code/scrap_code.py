
def calculate_covariance(context_dict, baroni_set, ft, window):
    covariance = {}

    for word in tqdm(baroni_set):
        total = np.zeros((100,100))

        for c_word in context_dict[word]:
            total += np.outer((ft.get_word_vector(c_word) - ft.get_word_vector(word)), (ft.get_word_vector(c_word) - ft.get_word_vector(word)))

        covariance[word] = (total / (len(context_dict[word]) * window))

    return covariance

def create_word_covariance_matrix(embed_averages, i, word):
    # Extract the mean and variance tensors for the word at index i
    mean_tensor = embed_averages._sum[i] / embed_averages._counts[i]
    var_tensor = (embed_averages._ssq[i] - mean_tensor ** 2) / embed_averages._counts[i]

    # Stack the mean and variance tensors along the second dimension
    mean_var_tensor = torch.stack([mean_tensor, var_tensor], dim=1)

    # Calculate the covariance matrix for the word by taking the outer product of the mean_var_tensor
    # with itself
    cov_matrix = mean_var_tensor.t() @ mean_var_tensor

    return cov_matrix

word_cov_matrices = {}
for i in range(len(embavg._counts)):
    word = vocab._id_to_tok[i]
    word_cov_matrix = create_word_covariance_matrix(embavg, i, word)
    word_cov_matrices[word] = [word_cov_matrix, embavg._sum[i]]