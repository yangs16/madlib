/* ----------------------------------------------------------------------- *//**
 *
 * @file newplda.cpp
 *
 * @brief Functions for Parallel Latent Dirichlet Allocation
 *
 * @date Dec 13, 2012
 *//* ----------------------------------------------------------------------- */


#include <dbconnector/dbconnector.hpp>
#include <math.h>
#include "newplda.hpp"

namespace madlib {
namespace modules {
namespace newplda {

using madlib::dbconnector::postgres::madlib_get_typlenbyvalalign;
typedef struct __type_info{
    Oid oid;
    int16_t len;
    bool    byval;
    char    align;

    __type_info(Oid oid):oid(oid)
    {
        madlib_get_typlenbyvalalign(oid, &len, &byval, &align);
    }
} type_info;
type_info INT4TI(INT4OID);

/**
 * @brief Get the min value of an array - for parameter checking
 **/
static int32_t __min(ArrayHandle<int32_t> ah){
    const int32_t * array = ah.ptr();
    int32_t size = ah.size();
    int32_t min = array[0];
    for(int32_t i = 1; i < size; i++){
        if(array[i] < min)
            min = array[i];
    }
    return min;
}

/**
 * @brief Get the max value of an array - for parameter checking
 **/
static int32_t __max(ArrayHandle<int32_t> ah){
    const int32_t * array = ah.ptr();
    int32_t size = ah.size();
    int32_t max = array[0];
    for(int32_t i = 1; i < size; i++){
        if(array[i] > max)
            max = array[i];
    }
    return max;
}

/**
 * @brief Log_Beta function
 **/
static double __log_beta(double alpha, int32_t len){
	return len * lgamma(alpha) - lgamma(len * alpha);
}

/**
 * @brief Log_Multinomial_Beta function
 **/
static double __log_multi_beta(double * pTopics, int32 len){
	double res = 0.0;
	double sum = 0.0;
	for(int32 i = 0; i < len; i++){
		res += lgamma(pTopics[i]);
		sum += pTopics[i];
	}
	res -= lgamma(sum);
	return res;
}

/**
 * @brief SFunc for Compute the log likelihood of the LDA model
 **/
AnyType newplda_log_likelihood_sfunc::run(AnyType & args)
{
    double state = args[0].getAs<double>();
    double hyper = args[2].getAs<double>();

    if(hyper < 0)
        throw std::invalid_argument("Invalid argument - hyper.");

    ArrayHandle<int32_t> topic_count = args[1].getAs<ArrayHandle<int32_t> >();
    size_t topic_num = topic_count.size();
    if(topic_num < 1)
        return state;
    
    if(__min(topic_count) < 0)
        throw std::invalid_argument("Invalid values in topic_count.");
    
    double * tmp = new double[topic_num];
    for(size_t i = 0; i < topic_num; i++)
        tmp[i] = topic_count[i] + hyper;
	state += __log_multi_beta(tmp, topic_num);
	state -= __log_beta(hyper, topic_num);
    delete[] tmp;

    return state;
}

/**
 * @brief PreFunc for Compute the log likelihood of the LDA model
 **/
AnyType newplda_log_likelihood_prefunc::run(AnyType & args)
{
    double state1 = args[0].getAs<double>();
    double state2 = args[1].getAs<double>();
    return state1 + state2;
}

/**
 * @brief This function samples a new topic for a word in a document based on
 * the topic counts computed on the rest of the corpus. This is the core
 * function in the Gibbs sampling inference algorithm for LDA. 
 * @param topic_num     The number of topics
 * @param topic         The current topic assignment to a word
 * @param count_d_z     The document topic counts
 * @param count_w_z     The word topic counts
 * @param count_z       The corpus topic counts 
 * @param alpha         The Dirichlet parameter for the per-doc topic
 *                      multinomial 
 * @param beta          The Dirichlet parameter for the per-topic word
 *                      multinomial
 * @return retopic      The new topic assignment to the word
 * @note The topic ranges from 0 to topic_num - 1. 
 **/
static int32_t __newplda_gibbs_sample(
    int32_t topic_num, int32_t topic, const int32_t * count_d_z, const int32_t * count_w_z,
    const int32_t * count_z, double alpha, double beta) 
{
    /* The cumulative probability distribution of the topics */
    double * topic_prs = new double[topic_num]; 

    /* Calculate topic (unnormalised) probabilities */
    double total_unpr = 0;
    for (int32_t i = 0; i < topic_num; i++) {
        int32_t nwz = count_w_z[i];
        int32_t ndz = count_d_z[i];
        int32_t nz = count_z[i];

        /* Adjust the counts to exclude current word's contribution */
        if (i == topic) {
            nwz--;
            ndz--;
            nz--;
        }

        /* Compute the probability */
        double unpr = (ndz + alpha) * (nwz + beta) / (nz + topic_num * beta);
        total_unpr += unpr;
        topic_prs[i] = total_unpr;
    }

    /* Normalise the probabilities */
    for (int32_t i = 0; i < topic_num; i++)
        topic_prs[i] /= total_unpr;

    /* Draw a topic at random */
    double r = drand48();
    int32_t retopic = 0;
    while (true) {
        if (retopic == topic_num - 1 || r < topic_prs[retopic])
            break;
        retopic++; 
    }

    delete[] topic_prs;
    return retopic;
}


/**
 * @brief This function learns the topics of words in a document and is the
 * main step of a Gibbs sampling iteration.
 * @param args[0]           The current topics assigned to a word in a document
 * @param args[1]           The per-document topic counts
 * @param args[2]           The per-word topic counts
 * @param args[3]           The corpus-level topic counts
 * @return                  The updated topic assignments
 **/
AnyType newplda_gibbs_sample::run(AnyType & args)
{
    MutableArrayHandle<int32_t> topic_assignment = args[0].getAs<MutableArrayHandle<int32_t> >();
    ArrayHandle<int32_t> doc_topic = args[1].getAs<ArrayHandle<int32_t> >();
    ArrayHandle<int32_t> word_topic = args[2].getAs<ArrayHandle<int32_t> >();
    ArrayHandle<int32_t> corpus_topic = args[3].getAs<ArrayHandle<int32_t> >();
    double alpha = args[4].getAs<double>();
    double beta = args[5].getAs<double>();
    int32_t topic_num = word_topic.size();

    if((size_t)topic_num != corpus_topic.size())
        throw std::invalid_argument(
            "Dimensions mismatch - word_topic.size() != corpus_topic.size().");
    if(alpha <= 0)
        throw std::invalid_argument("Invalid argument - alpha.");
    if(beta <= 0)
        throw std::invalid_argument("Invalid argument - beta.");
    if(__min(topic_assignment) < 0 || __max(topic_assignment) >= topic_num)
        throw std::invalid_argument(
            "Invalid values in topic_assignment.");

    for(size_t i = 0; i < topic_assignment.size(); i++){
        int32_t topic = topic_assignment[i];
        int32_t retopic = __newplda_gibbs_sample(
            topic_num, topic, doc_topic.ptr(), word_topic.ptr(),
            corpus_topic.ptr(), alpha, beta);
            topic_assignment[i] = retopic;
    }

    return topic_assignment;
}

/**
 * @brief This function assigns topics to words in a document randomly and
 * returns the topic counts and topic assignments.
 * @param args[0]   The word count in the documents
 * @param args[1]   The number of topics
 * @result          The topic counts and topic assignments 
 *                  (length = topic_num + word_count)
 **/
AnyType newplda_random_assign::run(AnyType & args)
{
    int32_t word_count = args[0].getAs<int32_t>();
    int32_t topic_num = args[1].getAs<int32_t>();

    if(word_count < 1)
        throw std::invalid_argument( "Invalid argument - word_count.");
    if(topic_num < 1)
        throw std::invalid_argument( "Invalid argument - topic_num.");

    MutableArrayHandle<int32_t> topic_assignment(
        construct_array(
            NULL, word_count, INT4TI.oid, INT4TI.len, INT4TI.byval,
            INT4TI.align));

    for(int32_t i = 0; i < word_count; i++){
        int32_t topic = rand() % topic_num;
        topic_assignment[i] = topic;  
    }

    return topic_assignment;
}

/**
 * @brief This function is the sfunc for the aggregator computing the topic
 * counts. It scans the topic assignments and updates the topic counts.
 * @param args[0]   The state variable, current topic counts
 * @param args[1]   The current topics as assigned to a word in a document
 * @return          The updated state
 **/
AnyType newplda_count_topic_sfunc::run(AnyType & args)
{
    if(!(args.getFCInfo()->context && IsA(args.getFCInfo()->context, AggState)))
        throw std::runtime_error(
            "This function is not used as part of an aggregator.");

    if(args[2].isNull())
        throw std::invalid_argument("Null argument - topic_num.");
    if(args[1].isNull())
        return args[0];

    ArrayHandle<int32_t> topic_assignment = 
        args[1].getAs<ArrayHandle<int32> >();
    int32_t topic_num = args[2].getAs<int32_t>();
    if(topic_num < 0)
        throw std::invalid_argument("Invalid argument - topic_num.");
    if(__min(topic_assignment) < 0 || __max(topic_assignment) >= topic_num)
        throw std::invalid_argument("Invalid values in topic_assignment.");

    MutableArrayHandle<int32_t> state(NULL);
    if(args[0].isNull()){
        state = construct_array(
            NULL,  topic_num, INT4TI.oid, INT4TI.len, INT4TI.byval,
            INT4TI.align);
    } else {
        state = args[0].getAs<MutableArrayHandle<int32_t> >();
    }

    for(size_t i = 0; i < topic_assignment.size(); i++){
            int32_t topic = topic_assignment[i];
            state[topic]++;
    }

    return state;
}

/**
 * @brief This function is the prefunc for the aggregator computing the
 * topic counts.
 * @param args[0]   The state variable, local topic counts
 * @param args[1]   The state variable, local word topic counts
 * @return          The merged state, element-wise sum of two local states
 **/
AnyType newplda_count_topic_prefunc::run(AnyType & args)
{
    MutableArrayHandle<int32_t> state1 = args[0].getAs<MutableArrayHandle<int32_t> >();
    ArrayHandle<int32_t> state2 = args[1].getAs<ArrayHandle<int32_t> >();

    if(state1.size() != state2.size())
        throw std::invalid_argument("Invalid dimension.");

    for(size_t i = 0; i < state1.size(); i++)
        state1[i] += state2[i];
    
    return state1;
}
/**
 * @breif This function is the sfunc of the aggregator getting the first value
 * of a column in a segment. 
 * @param arg[0]    The state variable, null or the first value 
 * @param arg[1]    The value in the column in a row
 * @return          Updated state 
 **/
AnyType newplda_first::run(AnyType & args)
{
    if (args[0].isNull()) 
        return args[1]; 
    else 
        return args[0]; 
}

}
}
}
