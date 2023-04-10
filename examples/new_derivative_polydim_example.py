from sympy import symbols,Matrix,diff
import numpy as np
import itertools


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

def delete_duplicates(x):
  return list(dict.fromkeys(x))



def poly_model(dim=2,order=2):
    x=[symbols('x'+str(i)) for i in range(dim)]
    terms=[]
    for n in range(1,order+1):
        terms.append(create_poly_terms(x,degree=n))
    terms.reverse()
    terms.append([x[1]**0])
    flattened_list=list(itertools.chain(*terms))
    return delete_duplicates(flattened_list)


def create_subs(symbol,point):
    subs=[]
    for i,symb in enumerate(symbol):
        subs.append((symb,point[i]))
    return subs



def main_system(model,der_grid):
    dim=len(der_grid[1])
    x=[symbols('x'+str(i)) for i in range(dim)]
    system=[]
    for point in der_grid:
        subs_model=[term.subs(create_subs(x,point)) for term in model]
        system.append(subs_model)
    return system

from copy import copy

def matrix_replace_row(matrix,row,rown):
    matrix1=copy(matrix)
    matrix1.row_del(rown)
    matrix1=matrix1.row_insert(rown, Matrix([row]))
    return matrix1




def lagrange_interp_expression(model,interp_grid):
    npts=len(interp_grid)
    sys=main_system(model,interp_grid)
    mat=Matrix(sys)
    det=mat.det()
    if det==0:
        print('Warning: We were unable to interpolate surfce at point {} with the current model. Please consider changing either model or adding points to der_grid. Zeros are returned.'.format(interp_grid[0]))
        return 0
    interp_expression=0
    f=[symbols('f'+str(i)) for i in range(npts)]
    for i in range(npts):
        mat1=matrix_replace_row(mat,model,0)
        det1=mat1.det()
        interp_expression+=f[i]*det1/det
    return interp_expression

def lagrange_interp_weights(model,interp_grid):
    npts=len(interp_grid)
    sys=main_system(model,interp_grid)
    mat=Matrix(sys)
    det=mat.det()
    if det==0:
        print('Warning: We were unable to interpolate surfce at point {} with the current model. Please consider changing either model (order) or adding points to der_grid. Zeros are returned.'.format(interp_grid[0]))
        return [0 for _ in range(npts)]
    weights=[]
    for _ in range(npts):
        mat1=matrix_replace_row(mat,model,0)
        det1=mat1.det()
        weights.append(det1/det)
    return weights






#expr=lagrange_interp_expression(model,der_grid)

#weights=lagrange_interp_weights(model,der_grid)

#print(expr)

#print(diff(expr,symbols('x0')))

#diff_weights=[diff(weight,symbols('x0')) for weight in weights]

#print(diff_weights)

def distances(der_grid,x0):
    return np.linalg.norm(np.array(der_grid)-np.array(x0),axis=1)



def pick_points(der_grid,x0,npoints):
    dist=distances(der_grid,x0)
    print(dist)
    picked_points_number=np.argsort(dist)[:npoints]
    return picked_points_number

#model=poly_model(dim=2,order=1)

#der_grid=[[0,0],[0,1],[1,0],[1,1],[0,0.5],[0.5,0]]

#npoints=len(model)

#picked_points=pick_points(der_grid,[1,0],model)

#print(model)

#interp_grid=[der_grid[i] for i in picked_points]

#print(interp_grid)

#weights=lagrange_interp_weights(model,interp_grid)

#print(weights)

#diff_weights=[diff(weight,symbols('x0')) for weight in weights]

#print(diff_weights)

def compute_weigths(model,der_grid,comp_grid):
    weights=[]
    npoints=len(model)
    if npoints>len(der_grid):
        print('Number of differentiation points less than model parameters')
        return None
    for x0 in comp_grid:
        picked_points=pick_points(der_grid,x0,npoints)
        interp_grid=[der_grid[i] for i in picked_points]
        w=lagrange_interp_weights(model,interp_grid)
        weights.append(w)
    return weights

model=poly_model(dim=2,order=1)

der_grid=[[0,0],[0,1],[1,0],[1,1],[0,0.5],[0.5,0]]

comp_grid=copy(der_grid)

weights=compute_weigths(model,der_grid,comp_grid)

print(weights)
                


