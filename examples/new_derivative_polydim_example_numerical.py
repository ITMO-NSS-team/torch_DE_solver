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


def distances(der_grid,x0):
    return np.linalg.norm(np.array(der_grid)-np.array(x0),axis=1)



def pick_points(der_grid,x0,npoints):
    dist=distances(der_grid,x0)
    picked_points_number=np.argsort(dist)[:npoints]
    return picked_points_number

#model=poly_model(dim=2,order=1)

#print(model)

#der_grid=[[1/10*i,1/10*j] for i in range(10) for j in range(10)]

##interp_points=[der_grid[i] for i in pick_points(der_grid,der_grid[5],3)]

#interp_points=[[0,0],[0,1],[1,1]]

#print(interp_points)

#sys=np.array((main_system(model,interp_points)),dtype=np.float64)

#print(sys)

#print(np.linalg.solve(sys,np.array([1,0,0])))


def lagrange_interp_weights_n(model,interp_grid):
    npts=len(interp_grid)
    sys=np.array(main_system(model,interp_grid),dtype=np.float64)
    #mat=Matrix(sys)
    #det=mat.det()
    if np.linalg.det(sys)==0:
        print('Warning: We were unable to interpolate surfce at point {} with the current model. Please consider changing either model (order) or adding points to der_grid. Zeros are returned.'.format(interp_grid[0]))
        return [[0 for _ in range(npts)] for _ in range(npts)]
    weights=[]
    for i in range(npts):
        point_characteristic=np.zeros(npts)
        point_characteristic[i]=1
        coeffs=np.linalg.solve(sys,point_characteristic)
        weights.append(model*coeffs)
    return weights


def compute_weigths(model,der_grid,comp_grid):
    weights=[]
    npoints=len(model)
    if npoints>len(der_grid):
        print('Number of differentiation points less than model parameters')
        return None
    for x0 in comp_grid:
        picked_points=pick_points(der_grid,x0,npoints)
        interp_grid=[der_grid[i] for i in picked_points]
        w=lagrange_interp_weights_n(model,interp_grid)
        weights.append(w)
    return weights


                

def compute_derivative(weights, axes=[0]):
    x=[symbols('x'+str(i)) for i in range(max(axes)+1)]
    deriv=weights
    for axis in axes:
        deriv=[[[diff(characteristic,x[axis]) for characteristic in model] for model in point] for point in deriv]
    return deriv

model=poly_model(dim=2,order=3)

der_grid=[[1/10*i,1/10*j] for i in range(10) for j in range(10)]

from copy import copy

comp_grid=copy(der_grid)

weights=compute_weigths(model,der_grid,comp_grid)

#print(weights)

der=compute_derivative(weights,axes=[0,1])

#print(list(der))


#from copy import copy

#def matrix_replace_row(matrix,row,rown):
#    matrix1=copy(matrix)
#    matrix1.row_del(rown)
#    matrix1=matrix1.row_insert(rown, Matrix([row]))
#    return matrix1




#def lagrange_interp_expression(model,interp_grid):
#    npts=len(interp_grid)
#    sys=main_system(model,interp_grid)
#    mat=Matrix(sys)
#    det=mat.det()
#    if det==0:
#        print('Warning: We were unable to interpolate surfce at point {} with the current model. Please consider changing either model or adding points to der_grid. Zeros are returned.'.format(interp_grid[0]))
#        return 0
#    interp_expression=0
#    f=[symbols('f'+str(i)) for i in range(npts)]
#    for i in range(npts):
#        mat1=matrix_replace_row(mat,model,0)
#        det1=mat1.det()
#        interp_expression+=f[i]*det1/det
#    return interp_expression

#def lagrange_interp_weights(model,interp_grid):
#    npts=len(interp_grid)
#    sys=main_system(model,interp_grid)
#    mat=Matrix(sys)
#    det=mat.det()
#    if det==0:
#        print('Warning: We were unable to interpolate surfce at point {} with the current model. Please consider changing either model (order) or adding points to der_grid. Zeros are returned.'.format(interp_grid[0]))
#        return [0 for _ in range(npts)]
#    weights=[]
#    for _ in range(npts):
#        mat1=matrix_replace_row(mat,model,0)
#        det1=mat1.det()
#        weights.append(det1/det)
#    return weights



#def distances(der_grid,x0):
#    return np.linalg.norm(np.array(der_grid)-np.array(x0),axis=1)



#def pick_points(der_grid,x0,npoints):
#    dist=distances(der_grid,x0)
#    picked_points_number=np.argsort(dist)[:npoints]
#    return picked_points_number



#def compute_weigths(model,der_grid,comp_grid):
#    weights=[]
#    npoints=len(model)
#    if npoints>len(der_grid):
#        print('Number of differentiation points less than model parameters')
#        return None
#    for x0 in comp_grid:
#        picked_points=pick_points(der_grid,x0,npoints)
#        interp_grid=[der_grid[i] for i in picked_points]
#        w=lagrange_interp_weights(model,interp_grid)
#        weights.append(w)
#    return weights


                

#def compute_derivative(weights, axes=[0]):
#    x=[symbols('x'+str(i)) for i in range(max(axes)+1)]
#    deriv=weights
#    for axis in axes:
#        deriv=[[diff(weight,x[axis]) for weight in point] for point in deriv]
#    return deriv

#model=poly_model(dim=2,order=3)

#der_grid=[[1/10*i,1/10*j] for i in range(10) for j in range(10)]

#comp_grid=copy(der_grid)

#weights=compute_weigths(model,der_grid,comp_grid)

#print(weights)

#der=compute_derivative(weights,axes=[0,1])

#print(list(der))
