#include <iostream>
#include <vector>
#include <map>
#include <Eigen/Dense>

/**
 * This file implements the primal dual schema for weighted set cover.
 * It gives a guarantee of $f$ if each element is in at most $f$ sets.
 * E.g., for Weighted Vertex Cover it gives a 2-approximation.
 */

#define EPSILON 0.0001 //We are working with doubles and the tightness of a linear inequality can become slightly inaccurate.

// use 0,1,...,(nr_elements-1) as indices for elements
// Sets are indexed by 0,...,(sets.size()-1)
struct Instance
{
    const size_t nr_elements;
    std::vector<std::vector<int>> sets;
    std::vector<double> costs;

    Instance(size_t nr_elements): nr_elements{nr_elements}{}

    void add_set(double cost, std::vector<int> covered_elements){
        sets.push_back(covered_elements);
        costs.push_back(cost);
    }

    void verify(){
        assert(costs.size()==sets.size());
        for(auto s: sets){
            for(auto e: s){
                assert(e>=0);
                assert(e<nr_elements);
            }
        }
    }
};

std::vector<int> solve(Instance& instance){
    instance.verify();

    auto nr_sets = instance.sets.size();

    // (e,s)=1 iff e is covered by s
    Eigen::MatrixXd A(instance.nr_elements, instance.sets.size());
    for(int c=0; c< nr_sets; ++c){
        for(int r: instance.sets[c]){
            A(r,c) = 1;
        }
    }

    //Copy cost-std::vector to a Eigen::Vector
    Eigen::VectorXd c(nr_sets);
    for(int i=0; i<nr_sets; ++i) c[i]=instance.costs[i];

    Eigen::VectorXd sum{A.cols()}; // sum <= c
    for(int e=0; e<instance.nr_elements; ++e){ //go over all elements
        auto covered_by_sets = A.row(e).transpose(); //the vector describing by which sets the element is covered
        auto gap = c-sum; //gap until dual constraints become tight

        //check how far we can increase the dual variable y_e
        auto max_possible_incr = std::numeric_limits<double>::infinity();
        for(int i=0; i<nr_sets; ++i){
            if(covered_by_sets[i]<EPSILON) continue;
            max_possible_incr = std::min(max_possible_incr, gap[i]/covered_by_sets[i]);
        }

        //if we can increase it arbitrarily, the dual is unbounded and thus the primal is infeasible
        if(max_possible_incr==std::numeric_limits<double>::infinity()){
            std::cerr << "Infeasible!" << std::endl;
            return {};
        }

        //increase the dual maximally
        sum+= max_possible_incr*covered_by_sets;
    }

    //return all sets for which the constraint in the dual is tight.
    std::vector<int> set_cover;
    for(int s=0; s<nr_sets; ++s){
        if( std::fabs(c[s]-sum[s])< EPSILON) set_cover.push_back(s);
    }

    //TODO: The algorithm possibly adds redundant sets which could be removed here. Try to remove the expensive sets first.

    return set_cover;
}

int main()
{
    Instance instance{5};
    instance.add_set(50, {0,1});
    instance.add_set(2, {1, 2,3});
    instance.add_set(3, {3,4});
    instance.add_set(2, {4,0});
    std::cout << "Using sets: ";
    for(auto s: solve(instance)) std::cout << "S_"<<s<<"\t";
    std::cout << std::endl;

    return 0;
}