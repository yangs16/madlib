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
    int32_t topic_num, int32_t topic, int32_t * count_d_z, int32_t * count_w_z,
    int32_t * count_z, float8 alpha, float8 beta) 
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
    float8 r = drand48();
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
 * @brief This function predicts the topics of words in a document given the
 * learned topic models. The learned topic modesl are passed to this function
 * in the first call and then transfered to the rest calls through
 * args.mSysInfo->user_fctx for efficiency. 
 * @param args[0]           The word count in the document
 * @param args[1]           The unique words in the documents
 * @param args[2]           The counts of each unique words
 * @param args[3]           The topic counts and the topic assignments in the
 *                          document
 * @param args[4]           The word topic counts, not null for the first call
 *                          in each segment, but null for the rest calls for
 *                          efficiency, refer to the tricks in the join
 *                          operation in the sql calling this function
 * @param args[5]           The corpus topic counts, not null for the first
 *                          call in each segment, but null for the rest calls
 *                          for efficiency, refer to the tricks in the join
 *                          operation in the sql calling this function
 * @param args[6]           The Dirichlet parameter for per-document topic
 *                          multinomial, i.e. alpha
 * @param args[7]           The Dirichlet parameter for per-topic word
 *                          multinomial, i.e. beta
 * @param args[8]           The size of vocabulary
 * @param args[9]           The number of topics
 * @param args[10]          The nunber of iterations (e.g. 20)
 * @return                  The predicted topic counts and topic assignments for
 *                          the document
 **/
AnyType newplda_gibbs_pred::run(AnyType & args)
{
    int32_t word_count = args[0].getAs<int32_t>();
    ArrayHandle<int32_t> words = args[1].getAs<ArrayHandle<int32_t> >();
    ArrayHandle<int32_t> counts = args[2].getAs<ArrayHandle<int32_t> >();
    ArrayHandle<int32_t> topics = args[3].getAs<ArrayHandle<int32_t> >();

    double alpha = args[6].getAs<double>();
    double beta = args[7].getAs<double>();
    int32_t voc_size = args[8].getAs<int32_t>();
    int32_t topic_num = args[9].getAs<int32_t>();
    int32_t iter_num = args[10].getAs<int32_t>();

    int32_t __state_size = (voc_size + 1) * topic_num;
    if (!args.getSysInfo()->user_fctx)
    {
        if(args[4].isNull() || args[5].isNull()){
            throw std::domain_error(
                "The parameters word_topic and corpus_topic should not be \
                null for the first call of newplda_gibbs_pred"); 
        }
        ArrayHandle<int32_t> word_topic = args[4].getAs<ArrayHandle<int32_t> >();
        ArrayHandle<int32_t> corpus_topic = args[5].getAs<ArrayHandle<int32_t> >();

        args.getSysInfo()->user_fctx =
            MemoryContextAllocZero(
                    args.getSysInfo()->cacheContext,
                    __state_size * sizeof(int32_t));

        int32_t * state = (int32_t *) args.getSysInfo()->user_fctx;
        memcpy(state, word_topic.ptr(), (voc_size * topic_num) * sizeof(int32_t));
        memcpy(state + voc_size * topic_num, corpus_topic.ptr(), topic_num * sizeof(int32_t));
    }
    int32_t * state = (int32_t *) args.getSysInfo()->user_fctx;
    if(NULL == state){
        throw std::runtime_error("The args.mSysInfo->user_fctx is null.");
    }

    MutableArrayHandle<int32_t> outarray(
        construct_array(
            NULL, topic_num + word_count, INT4TI.oid, INT4TI.len, INT4TI.byval,
            INT4TI.align));
    memcpy(outarray.ptr(), topics.ptr(), (topic_num + word_count) * sizeof(int32_t));

    int32_t unique_word_count = words.size();
    for(int it = 0; it < iter_num; it++){
        int32_t word_index = topic_num;
        for(int32_t i = 0; i < unique_word_count; i++) {
            int32_t wordid = words[i];
            for(int32_t j = 0; j < counts[i]; j++){
                int32_t topic = outarray[word_index];
                int32_t retopic = __newplda_gibbs_sample(
                    topic_num, topic, outarray.ptr(), state + wordid * topic_num, 
                    state + voc_size * topic_num, alpha, beta);

                outarray[word_index] = retopic;
                outarray[topic]--;
                outarray[retopic]++;
                word_index++;
            }
        }
    }

    return outarray;
}

/**
 * @brief This function learns the topics of words in a document and is the
 * main step of a Gibbs sampling iteration. The word topic counts and
 * corpus topic counts are passed to this function in the first call and
 * then transfered to the rest calls through args.mSysInfo->user_fctx for
 * efficiency. 
 * @param args[0]           The word count in the document
 * @param args[1]           The unique words in the documents
 * @param args[2]           The counts of each unique words
 * @param args[3]           The topic counts and the topic assignments in the
 *                          document
 * @param args[4]           The word topic counts, not null for the first call
 *                          in each segment, but null for the rest calls for
 *                          efficiency, refer to the tricks in the join
 *                          operation in the sql calling this function
 * @param args[5]           The corpus topic counts, not null for the first
 *                          call in each segment, but null for the rest calls
 *                          for efficiency, refer to the tricks in the join
 *                          operation in the sql calling this function
 * @param args[6]           The Dirichlet parameter for per-document topic
 *                          multinomial, i.e. alpha
 * @param args[7]           The Dirichlet parameter for per-topic word
 *                          multinomial, i.e. beta
 * @param args[8]           The size of vocabulary
 * @param args[9]           The number of topics
 * @return                  The updated topic counts and topic assignments for
 *                          the document
 **/
AnyType newplda_gibbs_train::run(AnyType & args)
{
    int32_t word_count = args[0].getAs<int32_t>();
    ArrayHandle<int32_t> words = args[1].getAs<ArrayHandle<int32_t> >();
    ArrayHandle<int32_t> counts = args[2].getAs<ArrayHandle<int32_t> >();
    ArrayHandle<int32_t> topics = args[3].getAs<ArrayHandle<int32_t> >();

    double alpha = args[6].getAs<double>();
    double beta = args[7].getAs<double>();
    int32_t voc_size = args[8].getAs<int32_t>();
    int32_t topic_num = args[9].getAs<int32_t>();

    int32_t __state_size = (voc_size + 1) * topic_num;

    if (!args.getSysInfo()->user_fctx)
    {
        if(args[4].isNull() || args[5].isNull()){
            throw std::domain_error(
                "The parameters word_topic and corpus_topic should not be null \
                for the first call of newplda_gibbs_train"); 
        }
        ArrayHandle<int32_t> word_topic = args[4].getAs<ArrayHandle<int32_t> >();
        ArrayHandle<int32_t> corpus_topic = args[5].getAs<ArrayHandle<int32_t> >();

        args.getSysInfo()->user_fctx =
            MemoryContextAllocZero(
                    args.getSysInfo()->cacheContext,
                    __state_size * sizeof(int32_t));
        elog(NOTICE, "after FCInfo()");

        int32_t * state = (int32_t *) args.getSysInfo()->user_fctx;
        memcpy(state, word_topic.ptr(), (voc_size * topic_num) * sizeof(int32_t));
        memcpy(state + voc_size * topic_num, corpus_topic.ptr(), topic_num * sizeof(int32_t));
        elog(NOTICE, "after memcpy");
    }

    int32_t * state = (int32_t *) args.getSysInfo()->user_fctx;
    if(NULL == state){
        throw std::runtime_error("The args.mSysInfo->user_fctx is null.");
    }

    MutableArrayHandle<int32_t> outarray(
        construct_array(
            NULL, topic_num + word_count, INT4TI.oid, INT4TI.len, INT4TI.byval,
            INT4TI.align));
    memcpy(outarray.ptr(), topics.ptr(), (topic_num + word_count) * sizeof(int32_t));

    int32_t unique_word_count = words.size();
    int32_t word_index = topic_num;
    for(int32_t i = 0; i < unique_word_count; i++) {
        int32_t wordid = words[i];
        for(int32_t j = 0; j < counts[i]; j++){
            int32_t topic = outarray[word_index];
            int32_t retopic = __newplda_gibbs_sample(
                topic_num, topic, outarray.ptr(), state + wordid * topic_num, 
                state + voc_size * topic_num, alpha, beta);
            outarray[word_index] = retopic;
            outarray[topic]--;
            outarray[retopic]++;

            state[voc_size * topic_num + topic]--;
            state[voc_size * topic_num + retopic]++;
            state[wordid * topic_num + topic]--;
            state[wordid * topic_num + retopic]++;
            word_index++;
        }
    }

    return outarray;
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
        throw std::invalid_argument(
            "The word count should be positive integer.");
    if(topic_num < 1)
        throw std::invalid_argument(
            "The topic number should be positive integer.");

    MutableArrayHandle<int32_t> outarray(
        construct_array(
            NULL, topic_num + word_count, INT4TI.oid, INT4TI.len, INT4TI.byval,
            INT4TI.align));

    for(int32_t i = 0; i < word_count; i++){
        int32_t topic = rand() % topic_num;
        outarray[topic] += 1;
        outarray[topic_num + i] = topic;  
    }

    return outarray;
}

/**
 * @brief This function is the sfunc for the aggregator computing the word
 * topic counts. It scans the topic assignments in a document and updates
 * the word topic counts.
 * @param args[0]   The state variable, current word topic counts
 *                  (length = voc_size * topic_num) 
 * @param args[1]   The unique words in the document (the wordid ranges from 0 to
 *                  voc_size - 1) 
 * @param args[2]   The counts of each unique word in the document
 *                  (sum(counts) = word_count) 
 * @param args[3]   The topic assignments in the document
 * @param args[4]   The size of vocabulary
 * @param args[5]   The number of topics 
 * @return          The updated state
 **/
AnyType newplda_count_word_topic::run(AnyType & args)
{
    int32_t voc_size = args[4].getAs<int32_t>();
    int32_t topic_num = args[5].getAs<int32_t>();

    MutableArrayHandle<int32_t> state(NULL);
    if(args[0].isNull()){
        state = construct_array(
            NULL, voc_size * topic_num, INT4TI.oid, INT4TI.len, INT4TI.byval,
            INT4TI.align);
    } else {
        state = args[0].getAs<MutableArrayHandle<int32_t> >();
    }

    ArrayHandle<int32_t> words = args[1].getAs<ArrayHandle<int32_t> >();
    ArrayHandle<int32_t> counts = args[2].getAs<ArrayHandle<int32_t> >();
    ArrayHandle<int32_t> topics = args[3].getAs<ArrayHandle<int32_t> >();

    int32_t unique_word_count = words.size();
    int32_t word_index = 0;
    for(int32_t i = 0; i < unique_word_count; i++){
        int32_t wordid = words[i];
        for(int32_t j = 0; j < counts[i]; j++){
            int32_t topic = topics[word_index];
            state[wordid * topic_num + topic]++;
            word_index++;
        }
    }

    return state;
}

/**
 * @brief This function is the prefunc for the aggregator computing the
 * word topic counts.
 * @param args[0]   The state variable, local word topic counts
 * @param args[1]   The state variable, local word topic counts
 * @return          The merged state, element-wise sum of two local states
 **/
AnyType newplda_array_add::run(AnyType & args)
{
    MutableArrayHandle<int32_t> inarray1 = args[0].getAs<MutableArrayHandle<int32_t> >();
    ArrayHandle<int32_t> inarray2 = args[1].getAs<ArrayHandle<int32_t> >();

    if(inarray1.size() != inarray2.size())
        throw std::invalid_argument("Invalid dimension.");

    for(uint32_t i = 0; i < inarray1.size(); i++)
        inarray1[i] += inarray2[i];
    
    return inarray1;
}

/**
 * @brief This udf treats a 1-d array as a 2-d array and then sum up the
 * embeded 1-d arrays. This will be used to compute the corpus-level topic
 * counts from the word topic counts.
 * @param args[0]   The input 1-d array
 * @param args[1]   The dimension of the embeded 1-d array
 * @return          The sum of the embeded 1-d arrays
**/
AnyType newplda_array_fold_add::run(AnyType & args)
{
    ArrayHandle<int32_t> inarray = args[0].getAs<ArrayHandle<int32_t> >();
    int32_t dim = args[1].getAs<int32_t>();
    int32_t size = inarray.size();

    if(dim < 0)
        throw std::invalid_argument("Invalid dimension.");
    
    if(dim >= size)
        return args[0];

    MutableArrayHandle<int32_t> outarray(
        construct_array(
            NULL, dim, INT4TI.oid, INT4TI.len, INT4TI.byval, INT4TI.align)); 

    for(int32_t i = 0; i < size; i++)
        outarray[i % dim] += inarray[i]; 
    
    return outarray;
}

/**
 * @breif This function is the sfunc of the aggregator getting the first value
 * of a column in a segment.
 * @param arg[0]    State variable, null or the first value
 * @param arg[1]    Value in the column in a row
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
