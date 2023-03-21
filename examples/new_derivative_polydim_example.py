from sympy import symbols

def create_poly_terms(variables,degree=1):
    if degree==1:
        return [v for v in variables]
    else:
        terms_list=[]
        for v in create_poly_terms(variables,degree=degree-1):
            for v1 in variables:
                terms_list.append(v*v1)
        return terms_list


#list_flatten =lambda nested_list: [item for sublist in nested_list for item in sublist]

import itertools

def poly_model(dim=2,order=2):
    x=[symbols('x'+str(i)) for i in range(dim)]
    terms=[]
    for n in range(1,order):
        terms.append(create_poly_terms(x,degree=n))
    terms.reverse()
    terms.append([1])
    return list(itertools.chain(*terms))


print(poly_model(dim=2,order=3))


der_grid=[[0,0],[0,1],[1,0]]