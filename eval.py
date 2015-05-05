    #   get the ngram probability for the validation set
    #   first load the relevant ngram model
    ngram_model = NgramModel.get(n, use_tree, ftr_use, feature_sizes,
                                 x_train, ngram_dir)
    #   then calculate validation set probability
    #   and also the probability of validation set with randomized
    #   conditioned term
    p_ngram = ngram_model.probability(x_valid)
    p_ngram_r = ngram_model.probability(x_valid_r)
    p_ngram_rr = ngram_model.probability(x_valid_rr)
    p_ngram_rrr = ngram_model.probability(x_valid_rrr)
    log.info("Ngram model ln(p_mean(x_valid)): %.3f, ln(p_mean(x_valid_r)):"
             " %3f, ln(p_mean(x_valid_rr)): %3f, ln(p_mean(x_valid_rrr)): %3f",
             np.log(p_ngram.mean()), np.log(p_ngram_r.mean()),
             np.log(p_ngram_rr.mean()), np.log(p_ngram_rrr.mean()))
    log.info("Ngram model p(x_valid) / p(x_valid_rand) mean: %.3f, "
             "p(x_valid) / p(rand) mean: %.3f,",
             (p_ngram / p_ngram_r).mean(),
             (p_ngram / p_ngram_rr).mean())
    log.info("Ngram model ll_mean(x_valid): %.3f, ll_mean(x_valid_r): %3f, "
             "ll_mean(x_valid_rr): %3f, ll_mean(x_valid_rrr): %3f",
             np.log(p_ngram).mean(), np.log(p_ngram_r).mean(),
             np.log(p_ngram_rr).mean(), np.log(p_ngram_rrr).mean())
    log.info("Ngram model ll(x_valid) / ll("
             "x_valid_rand) mean: %.3f, ll(x_valid) / ll(rand) mean: %.3f,",
             (np.log(p_ngram) - np.log(p_ngram_r)).mean(),
             (np.log(p_ngram) - np.log(p_ngram_rr)).mean())
    