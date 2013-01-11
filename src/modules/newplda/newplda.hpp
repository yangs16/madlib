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

DECLARE_UDF(newplda, newplda_transpose)
DECLARE_SR_UDF(newplda, newplda_unnest)
