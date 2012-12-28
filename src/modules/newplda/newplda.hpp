/* ----------------------------------------------------------------------- *//**
 *
 * @file newplda.hpp
 *
 * @brief Parallel Latent Dirichlet Allocation
 *
 *//* ----------------------------------------------------------------------- */

DECLARE_UDF(newplda, newplda_random_assign)
DECLARE_UDF(newplda, newplda_gibbs_sample)

DECLARE_UDF(newplda, newplda_count_topic_sfunc)
DECLARE_UDF(newplda, newplda_count_topic_prefunc)

DECLARE_UDF(newplda, newplda_log_likelihood_sfunc)
DECLARE_UDF(newplda, newplda_log_likelihood_prefunc)

DECLARE_UDF(newplda, newplda_first)
