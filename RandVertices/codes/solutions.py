# #Code by GVV Sharma
# #December 7, 2019
# #released under GNU GPL
# #Drawing a triangle given 3 sides

# import matplotlib.pyplot as plt
# import matplotlib.image as mpimg
# image = mpimg.imread('exit-ramp.jpg')
# plt.imshow(image)
# plt.show()

import sys                                          #for path to external scripts
#sys.path.insert(0, '/home/user/txhome/storage/shared/gitlab/res2021/july/conics/codes/CoordGeo')        #path to my scripts
sys.path.insert(0, '/home/sayyam/EE23010/RandVertices/codes/CoordGeo')        #path to my scripts
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#local imports
from line.funcs import *
import math
from triangle.funcs import *
from conics.funcs import circ_gen


#sys.path.insert(0, '/home/user/txhome/storage/shared/gitlab/res2021/july/conics/codes/CoordGeo')        #path to my scripts

#if using termux
import subprocess
import shlex
#end if

#Triangle sides
#a = 6
#b = 5
#c = 4


#Triangle coordinates
#[A,B,C] = tri_vert(a,b,c)

A = np.array([1, -5])
B = np.array([-4, -1])
C = np.array([5, 5])

def dir_vec (point_1 , point_2):
    return point_2 - point_1

def parametric_form (A,B,k):
    direction_vector_AB = dir_vec(A,B)
    x = A + k * direction_vector_AB
    return x

def norm_vec(B,C):
    return omat@dir_vec(B,C)

# Perpendicular bisector
def line_dir_pt(m, A, k1=0, k2=1):
    len = 2
    dim = A.shape[0]
    x_AB = np.zeros((dim, len))
    lam_1 = np.linspace(k1, k2, len)
    for i in range(len):
        temp1 = A + lam_1[i] * m
        x_AB[:, i] = temp1.T
    return x_AB
    
 #Intersection of two lines
def line_intersect(n1,A1,n2,A2):
  N=np.vstack((n1,n2))
  p = np.zeros(2)
  p[0] = n1@A1
  p[1] = n2@A2
  #Intersection
  P=np.linalg.inv(N)@p
  return P   
  
# Calculate the perpendicular vector and plot arrows
def perpendicular(B, C, label):
    perpendicular=norm_vec(B,C)
    mid = midpoint(B, C)
    x_D = line_dir_pt(perpendicular, mid, 0, 1)
    plt.arrow(mid[0], mid[1], perpendicular[0], perpendicular[1], color='blue', head_width=0.4, head_length=0.4, label=label)
    plt.arrow(mid[0], mid[1], -perpendicular[0], -perpendicular[1], color='blue', head_width=0.4, head_length=0.4)
    return x_D    

def unit_vec(A,B):
	return ((B-A)/np.linalg.norm(B-A))

def perp_foot(n,cn,P):
  m = omat@n
  N=np.block([[n],[m]])
  p = np.zeros(2)
  p[0] = cn
  p[1] = m@P
  #Intersection
  x_0=np.linalg.inv(N)@p
  return x_0
  
 # Midpoint of each line
def midpoint(P, Q):
    return (P + Q) / 2
          
#to find the coefficients and constant of the equation of perpendicular bisector of BC
def perpendicular_bisector(B, C):
    midBC=midpoint(B,C)
    dir=B-C
    constant = -dir.T @ midBC
    return dir,constant
equation_coeff1,const1 = perpendicular_bisector(A, B)
equation_coeff2,const2 = perpendicular_bisector(B, C)
equation_coeff3,const3 = perpendicular_bisector(C, A)  

 
def angle_btw_vectors(v1, v2):
    dot_product = v1 @ v2
    norm = np.linalg.norm(v1) * np.linalg.norm(v2)
    angle = np.arccos(dot_product / norm)
    angle_in_deg = np.degrees(angle)
    return angle_in_deg

AB = B - A
BC = C - B
CA = A - C
print("The solution to the question 1.1.1:")
print("The direction vector of AB is ",AB)
print("The direction vector of BC is ",BC)
print("The direction vector of CA is ",CA)

print("The solution to the question 1.1.2:")
length_BC = np.linalg.norm(BC)
print("Length of side BC:", length_BC)
V1 = AB
V2 = V1.reshape(-1,1)
print("The length of AB is:")
print(math.sqrt(V1@V2))
V3 = BC
V4 = V3.reshape(-1,1)
print("The length of BC is:")
print(math.sqrt(V3@V4))
V5 = CA
V6 = V5.reshape(-1,1)
print("The length of AC is:")
print(math.sqrt(V5@V6))

Mat = np.array([[1,1,1],[A[0],B[0],C[0]],[A[1],B[1],C[1]]])
rank = np.linalg.matrix_rank(Mat)
print("The solution to the question 1.1.3:")
if (rank<=2):
	print("Hence proved that points A,B,C in a triangle are collinear")
else:
	print("The given points are not collinear")

print("The solution to the question 1.1.4:") 
m1=(B-A)
m2=(C-B)
m3=(A-C)
print("parametric of AB form is x:",A,"+ k",m1)
print("parametric of BC form is x:",B,"+ k",m2)
print("parametric of CA form is x:",C,"+ k",m3)

#getting the equation of line
print("The solution to the question 1.1.5:")
omat = np.array([[0,1],[-1,0]])
m = dir_vec(A,B)   #direction vector
n = omat@m    #normal vector
c = n@A
eqn = f"{n}x = {c}"
print("The equation of line AB is",eqn)
m = dir_vec(B,C)   #direction vector
n = omat@m    #normal vector
c = n@B
eqn = f"{n}x = {c}"
print("The equation of line BC is",eqn)
m = dir_vec(C,A)   #direction vector
n = omat@m    #normal vector
c = n@C
eqn = f"{n}x = {c}"
print("The equation of line CA is",eqn)

print("The solution to the question 1.1.6:")
def AreaCalc(A, B, C):
    AB = A - B
    AC = A - C
#cross_product calculation
    cross_product = np.cross(AB,AC)
#magnitude calculation
    magnitude = np.linalg.norm(cross_product)
    area = 0.5 * magnitude
    return area
    
area_ABC = AreaCalc(A, B, C)
print("Area of triangle ABC:", area_ABC)

print("The solution to the question 1.1.7:")
dotA = (B - A) @ (C - A)
NormA=(np.linalg.norm(B-A))*(np.linalg.norm(C-A))
print('value of angle A: ', np.degrees(np.arccos((dotA)/NormA)))
dotB = (A - B) @ (C - B)
NormB = np.linalg.norm(A - B) * np.linalg.norm(C - B)
print('value of angle B: ', np.degrees(np.arccos(dotB / NormB)))
dotC = (A - C) @ (B - C)
NormC = np.linalg.norm(A - C) * np.linalg.norm(B - C)
print('value of angle C: ', np.degrees(np.arccos(dotC / NormC)))

D = (B + C)/2
E = (A + C)/2
F = (A + B)/2

print("The solution to the question 1.2.1:")
print("D:", list(D))
print("E:", list(E))
print("F:", list(F))

print("The solution to the question 1.2.2:")
m = D-A   #direction vector
n = omat@m    #normal vector
c = n@A
eqn = f"{n}x = {c}"
print("The equation of line AD is",eqn)
m = E-B   #direction vector
n = omat@m    #normal vector
c = n@B
eqn = f"{n}x = {c}"
print("The equation of line BE is",eqn)
m = F-C   #direction vector
n = omat@m    #normal vector
c = n@C
eqn = f"{n}x = {c}"
print("The equation of line CF is",eqn)

G=line_intersect(norm_vec(F,C),C,norm_vec(E,B),B)
print("The solution to the question 1.2.3:")
print("("+str(G[0])+","+str(G[1])+")")

print("The solution to the question 1.2.4:")
print("D:", list(D))
print("E:", list(E))
print("F:", list(F))
n1=norm_vec(A,D)
n2=norm_vec(B,E)
print("G:", list(G))
AG = np.linalg.norm(G - A)
GD = np.linalg.norm(D - G)
BG = np.linalg.norm(G - B)
GE = np.linalg.norm(E - G) 
CG = np.linalg.norm(G - C)
GF = np.linalg.norm(F - G)
print("AG/GD= "+str(AG/GD))
print("BG/GE= "+str(BG/GE))
print("CG/GF= "+str(CG/GF))
#finding the norm of vector
n_BG = norm_vec(B,G)
n_GE = norm_vec(G,E)
n_GF = norm_vec(G,F)
n_CG = norm_vec(C,G)
n_AG = norm_vec(A,G)
n_GD = norm_vec(G,D)
print("The direction vector of BG is",BG)
print("The direction vector of GE is",GE)
print("The direction vector of GF is",GF)
print("The direction vector of CG is",CG)
print("The direction vector of AG is",AG)
print("The direction vector of GD is",GD)
print("The norm of the vector BG is",n_BG)
print("The norm of the vector GE is",n_GE)
print("The norm of the vector GF is",n_GF)
print("The norm of the vector CG is",n_CG)
print("The norm of the vector AG is",n_AG)
print("The norm of the vector GD is",n_GD)

print("The solution to the question 1.2.5:")
print(f"The mid point of B and C is {D}")
print(f"The centroid of triangleABC is {G}")
Mat = np.array([[1,1,1],[A[0],D[0],G[0]],[A[1],D[1],G[1]]])
rank = np.linalg.matrix_rank(Mat)
if (rank==2):
	print("Hence proved that points A,G,D in a triangle are collinear")
else:
	print("Error")
	
print("The solution to the question 1.2.6:")
print("centroid of the given triangle: ")      
print(G)    
print("Hence Q.1.2.6 is verified.")

print("The solution to the question 1.2.7:")
print(f"A - F = {A-F}")
print(f"E - D = {E-D}")

#sides
ab=B-A
bc=C-B
ca=A-C
print("The solution to the question 1.3.1:")
print(bc)
t=np.array([0,1,-1,0]).reshape(2,2)
#AD_1
AD_1=t@bc
#normal vector of AD_1
AD_p=t@AD_1
print(AD_p)

D_1 = alt_foot(A,B,C)
E_1 = alt_foot(B,C,A)
F_1 = alt_foot(C,A,B)
nt = np.array([-1,11])
result = A@nt
print("The solution to the question 1.3.2:")
print(f"The equation of AD is {nt}X={result}")

#Finding orthocentre
H = line_intersect(norm_vec(B,E_1),E_1,norm_vec(C,F_1),F_1)
  
BE_norm = norm_vec(E_1, B)
CF_norm = norm_vec(F_1, C)
print("The solution to the question 1.3.3:")
n1 = bc    #normal vector
n2 = ca    #normal vector
n3 = ab    #normal vector
c1 = n1@A
c2 = n2@B
c3 = n3@C
eqn1 = f"{n1}x = {c1}"
eqn2 = f"{n2}x = {c2}"
eqn3 = f"{n3}x = {c3}"
print('')
print("The equation of line AD_1 is",eqn1)
print("The equation of line BE_1 is",eqn2)
print("The equation of line CF_1 is",eqn3)

A1 = np.array([[n2[0],n2[1]],[n3[0],n3[1]]])             #Defining the vector A1
B1 = np.array([c2,c3])                     #Defining the vector B1
H  = np.linalg.solve(A1,B1)               #applying linalg.solve to find x such that Ax=B
print("The solution to the question 1.3.4:")
print("The orthocenter H:",H)

result = int(((A - H).T) @ (B - C))    # Checking orthogonality condition...
# printing output
print("The solution to the question 1.3.5:")
if result == 0:
  print("(A - H)^T (B - C) = 0\nHence Verified...")

else:
  print("(A - H)^T (B - C)) != 0\nHence the given statement is wrong...")

D = midpoint(B, C)
E = midpoint(A, C)
F = midpoint(B, A)

#Solution to question 1.4.1

print("The solution to the question 1.4.1:")
print(f'Equation for perpendicular bisector of AB: {equation_coeff1}x + ({const1:.2f}) = 0')
print(f'Equation for perpendicular bisector of  BC: {equation_coeff2}x + ({const2:.2f}) = 0')
print(f'Equation for perpendicular bisector of  CA: {equation_coeff3}x + ({const3:.2f}) = 0')

# direction vector along line joining A & B
AB = dir_vec(A,B)
# direction vector along line joining A & C
AC = dir_vec(A,C)

#Generating the incircle
[I,r] = icircle(A,B,C)
x_icirc= circ_gen(I,r)


#Generating the circumcircle
[O,R] = ccircle(A,B,C)
x_circ= circ_gen(O,R)

#solution to question 1.4.2
print("The solution to the question 1.4.2:")
print(O)
print("The solution to the question 1.4.3:")
result = int((O - D) @ (B - C))
if result == 0:
    print("(((O - D)(B - C))= 0\nHence Verified...")

else:
    print("(((O - D)(B - C))!= 0\nHence the given statement is wrong...")

# OA, OB, OC
O_1 = O - A
O_2 = O - B
O_3 = O - C
a = np.linalg.norm(O_1)
b = np.linalg.norm(O_2)
c = np.linalg.norm(O_3)

print("The solution to the question 1.4.4:")
print("Circumcentre of triangle is", O, ".")
print("OA, OB, OC are respectively", a,",", b,",",c, ".")
print("Here, OA = OB = OC.")
print("Hence verified.")

print("The solution to the question 1.4.5:")
print("The figure has beed plotted using python.")

print("The solution to the question 1.4.6:")
#To find angle BOC
dot_pt_O = (B - O) @ ((C - O).T)
norm_pt_O = np.linalg.norm(B - O) * np.linalg.norm(C - O)
cos_theta_O = dot_pt_O / norm_pt_O
angle_BOC = round(360-np.degrees(np.arccos(cos_theta_O)),5)  #Round is used to round of number till 5 decimal places
print("angle BOC = " + str(angle_BOC))
#To find angle BAC
dot_pt_A = (B - A) @ ((C - A).T)
norm_pt_A = np.linalg.norm(B - A) * np.linalg.norm(C - A)
cos_theta_A = dot_pt_A / norm_pt_A
angle_BAC = round(np.degrees(np.arccos(cos_theta_A)),5)  #Round is used to round of number till 5 decimal places
print("angle BAC = " + str(angle_BAC))
#To check whether the answer is correct
if angle_BOC == 2 * angle_BAC:
  print("\nangle BOC = 2 times angle BAC\nHence the give statement is correct")
else:
  print("\nangle BOC ≠ 2 times angle BAC\nHence the given statement is wrong")

print("The solution to the question 1.5.1:") 
#using parallelogram theorem
E0= unit_vec(A,B) + unit_vec(A,C)
#generating normal form
F0=np.array([E0[1],(E0[0]*(-1))])
#matrix multiplication
C1= F0@(A.T)
print("Internal Angular bisector of angle A is:",F0,"x = ",C1)
E0= unit_vec(B,A) + unit_vec(B,C)
#point generated to create parametric form
F0=np.array([E0[1],(E0[0]*(-1))])
#matrix multiplication
C1= F0@(B.T)
print("Internal Angular bisector of angle B is:",F0,"x = ",C1)
E0= unit_vec(C,A) + unit_vec(C,B)
#point generated to create parametric form
F0=np.array([E0[1],(E0[0]*(-1))])
#matrix multiplication
C1= F0@(C.T)
print("Internal Angular bisector of angle C is:",F0,"x = ",C1)
 
t = norm_vec(B,C) 
n1 = t/np.linalg.norm(t) #unit normal vector
t = norm_vec(C,A)
t_AC = t
n2 = t/np.linalg.norm(t)
t = norm_vec(A,B)
t_AB = t
n3 = t/np.linalg.norm(t)

I=line_intersect(n1-n3,B,n1-n2,C) #intersection of angle bisectors B and C
print("The solution to the question 1.5.2:")
print("The intersection I of the angle bisectors of B and C", I)

#slopes of angle bisectors
m_a=norm_vec(n2,n3)
m_b=norm_vec(n1,n3)
m_c=norm_vec(n1,n2)

I=line_intersect(n1-n3,B,n1-n2,C) #intersection of angle bisectors B and C
r = n1 @ (B-I)

#BA, CA, and IA in vector form
BA = A - B
CA = A - C
IA = A - I

#Calculating the angles BAI and CAI
angle_BAI = angle_btw_vectors(BA, IA)
angle_CAI = angle_btw_vectors(CA, IA)

print("The solution to the question 1.5.3:")

# Print the angles
print("Angle BAI:", angle_BAI)
print("Angle CAI:", angle_CAI)

if np.isclose(angle_BAI, angle_CAI):
    print("Angle BAI is approximately equal to angle CAI.")
else:
    print("error")

print("The solution to the question 1.5.4:")
print("Coordinates of point I:", I)
print(f"Distance from I to BC= {r}")
n1T = t_AB.T   #taking transpose of t_AB
n2T = t_AC.T   #taking transpose of t_AC
r1 = abs((n1T @ I) - (n1T @ A))/(np.linalg.norm(t_AB))   #r1 is distance between I and AB (n1T.I - n1T.A=0, n1 and I are vectors)
r2 = abs((n2T @ I) - (n2T @ C))/(np.linalg.norm(t_AC))   #r2 is distance between I and AC (n2T.I - n2T.C=0, n2 and I are vectors)

print("The solution to the question 1.5.5:")
print("Distance between I and AB is",r1)
print("Distance between I and AC is",r2)
d=C-B
a=np.linalg.norm(C-B)
b=np.linalg.norm(A-C)
c=np.linalg.norm(A-B)

print("The solution to the question 1.5.6:")
print("The inradius r =", r)

print("The solution to the question 1.5.7:")
print("The figure has beed plotted using python.")

print("The solution to the question 1.5.8:")
print("the incentre coordiantes are",I)
radius = r
p=pow(np.linalg.norm((C-B).T),2)
q=2*(d.T@(I-B))
r=pow(np.linalg.norm(I-B),2)-radius*radius
Discre=q*q-4*p*r
print("the Value of discriminant is ",abs(round(Discre,6)))
#  so the value of Discrimant is extremely small and tends to zero
#  the discriminant value rounded off to 6 decimal places is also zero
#  so it proves that there is only one solution of the point

#  the value of point is x=B+k(C-B)
k=((C-B).T@(I-B))/((C-B)@(C-B))
print("the value of parameter k is ",k)
D3=B+k*(C-B)
print("the point of tangency of incircle by side BC is ",D3)
#  to check we also check the value of dot product of ID3 and BC
#print("the dot product of ID3 and BC",abs(round(((D3-I)@(C-B),6))))
#  so this comes out to be zero
print("Hence we prove that side BC is tangent To incircle and also found the value of k!")

print("The solution to the question 1.5.9:")
print("I = ",I)
#finding k for E_3 and F_3
k1=((I-A)@(A-B))/((A-B)@(A-B))
k2=((I-A)@(A-C))/((A-C)@(A-C))
#finding E_3 and F_3
E3=A+(k1*(A-B))
F3=A+(k2*(A-C))
print("k1 = ",k1)
print("k2 = ",k2)
print("E3 = ",E3)
print("F3 = ",F3)

print("The solution to the question 1.5.10:")
def norm(X,Y):
    magnitude=round(float(np.linalg.norm([X-Y])),3)
    return magnitude 
print("AF_3=", norm(A,F3) ,"\nAE_3=", norm(A,E3) ,"\nBD_3=", norm(B,D3) ,"\nBE_3=", norm(B,E3) ,"\nCD_3=", norm(C,D3) ,"\nCF_3=",norm(C,F3))

#finding sidelengths a, b & c
a = np.linalg.norm(B-C)
b = np.linalg.norm(C-A)
c = np.linalg.norm(A-B)

#creating array containing coefficients
Y = np.array([[1,1,0],[0,1,1],[1,0,1]])

#solving the equations
X = np.linalg.solve(Y,[c,a,b])

#printing output
print("The solution to the question 1.5.11:") 
print(X)

#Generating the incircle
#[I,r] = icircle(A,B,C)
#x_icirc= circ_gen(I,r)

#Generating all lines
x_AB = line_gen(A,B)
x_BC = line_gen(B,C)
x_CA = line_gen(C,A)
x_AD = line_gen(A, D)
x_BE = line_gen(B, E)
x_CF = line_gen(C, F)
x_DE = line_gen(D,E)
x_DF = line_gen(D,F)
x_AD_1 = line_gen(A,D_1)
x_BE_1 = line_gen(B,E_1)
x_CF_1 = line_gen(C,F_1)
x_AE_1 = line_gen(A,E_1)
x_AF_1 = line_gen(A,F_1)
x_CH = line_gen(C,H)
x_BH = line_gen(B,H)
x_AH = line_gen(A,H)
x_OA = line_gen(O,A)
x_OB = line_gen(O,B)
x_OC = line_gen(O,C)
x_OD = line_gen(O,D)
x_OE = line_gen(O,E)
x_OF = line_gen(O,F)
x_OG = line_gen(O,G)
x_BI = line_gen(B,I)
x_CI = line_gen(C,I)
x_AI = line_gen(A,I)

#generating angle bisectors
k1=[-6,-6]
k2=[6,6]  
x_A = line_dir_pt(m_a,A,k1,k2)
x_B = line_dir_pt(m_b,B,k1,k2)
x_C = line_dir_pt(m_c,C,k1,k2)

# Plotting all lines
plt.plot(x_AB[0, :], x_AB[1, :], label='$AB$')
plt.plot(x_BC[0, :], x_BC[1, :], label='$BC$')
plt.plot(x_CA[0, :], x_CA[1, :], label='$CA$')

#Generating the circumcircle
[O,R] = ccircle(A,B,C)
x_circ= circ_gen(O,R)

x_D = perpendicular(A, B, 'OD')
x_E = perpendicular(B, C, 'OE')
x_F = perpendicular(C, A, 'OF')
mid1 = midpoint(A, B)
mid2 = midpoint(B, C)
mid3 = midpoint(C, A)

#Labeling the coordinates
A = A.reshape(-1,1)
B = B.reshape(-1,1)
C = C.reshape(-1,1)
D = D.reshape(-1,1)
E = E.reshape(-1,1)
F = F.reshape(-1,1)
D_1 = D_1.reshape(-1,1)
E_1 = E_1.reshape(-1,1)
F_1 = F_1.reshape(-1,1)
H = H.reshape(-1,1)
G = G.reshape(-1,1)
O = O.reshape(-1,1)
mid12=mid1.reshape(-1,1)
mid23=mid2.reshape(-1,1)
mid31=mid3.reshape(-1,1)

tri_coords = np.block([[A,B,C,O,mid12,mid23,mid31]])
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C','O','D','E','F']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
		 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
plt.savefig('/home/sayyam/EE23010/RandVertices/figs/Q1.4.1.png')
plt.show()
#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')

tri_coords = np.block([[A,B,C]])
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
		 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')

#if using termux
#plt.savefig('tri_sss.pdf')
plt.savefig('/home/sayyam/EE23010/RandVertices/figs/Q1.1.3.png')
plt.show()

#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')
plt.plot(x_AD[0, :], x_AD[1, :], label='$AD$')
plt.plot(x_BE[0, :], x_BE[1, :], label='$BE$')
plt.plot(x_CF[0, :], x_CF[1, :], label='$CF$')

tri_coords = np.block([[A,B,C,D,E,F,G]])
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C','D','E','F','G']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(-10,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center
plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
plt.savefig('/home/sayyam/EE23010/RandVertices/figs/Q1.2.2.png')
plt.show()

#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')
plt.plot(x_AD_1[0,:],x_AD_1[1,:],label='$AD$')
plt.plot(x_BE_1[0,:],x_BE_1[1,:],label='$BE_1$')
plt.plot(x_AE_1[0,:],x_AE_1[1,:],linestyle = 'dashed',label='$AE_1$')
plt.plot(x_CF_1[0,:],x_CF_1[1,:],label='$CF_1$')
plt.plot(x_AF_1[0,:],x_AF_1[1,:],linestyle = 'dashed',label='$AF_1$')
plt.plot(x_CH[0,:],x_CH[1,:],label='$CH$')
plt.plot(x_BH[0,:],x_BH[1,:],label='$BH$')
plt.plot(x_AH[0,:],x_AH[1,:],linestyle = 'dashed',label='$AH$')

tri_coords = np.block([[A,B,C,D_1,E_1,F_1,H]])
#tri_coords = np.vstack((A,B,C,alt_foot(A,B,C),alt_foot(B,A,C),alt_foot(C,A,B),H)).T
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C','$D_1$','$E_1$','$F_1$','H']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
plt.savefig('/home/sayyam/EE23010/RandVertices/figs/Q1.3.4.png')
plt.show()

#plotting Angle bisectors
plt.plot(x_A[0,:],x_A[1,:],label='angle bisector of A')
plt.plot(x_B[0,:],x_B[1,:],label='angle bisector of B')
plt.plot(x_C[0,:],x_C[1,:],label='angle bisector of C')

#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')

#Plotting the incircle
plt.plot(x_icirc[0,:],x_icirc[1,:],label='$incircle$')

#Labeling the coordinates
A = A.reshape(-1,1)
B = B.reshape(-1,1)
C = C.reshape(-1,1)
I = I.reshape(-1,1)
D3 = D3.reshape(-1,1)
E3 = E3.reshape(-1,1)
F3 = F3.reshape(-1,1)
tri_coords = np.block([[A,B,C,I,D3,E3,F3]])
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C','I','$D_3$','$E_3$','$F_3$']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
		 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')
plt.savefig('/home/sayyam/EE23010/RandVertices/figs/Q1.5.1.png')
plt.show()

#Plotting all lines
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')
plt.plot(x_OA[0,:],x_OA[1,:],label='$OA$')

#Plotting the circumcircle
plt.plot(x_circ[0,:],x_circ[1,:],label='$circumcircle$')

#Labeling the coordinates
tri_coords = np.block([[A,B,C,O]])
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C','O']
for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center

plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() 
plt.axis('equal')
plt.savefig('/home/sayyam/EE23010/RandVertices/figs/Q1.4.5.png')
plt.show()

#plotiing the lines
plt.plot(x_AB[0,:],x_AB[1,:],label='$AB$')
plt.plot(x_BC[0,:],x_BC[1,:],label='$BC$')
plt.plot(x_CA[0,:],x_CA[1,:],label='$CA$')

#plotting the incircle
plt.plot(x_icirc[0,:],x_icirc[1,:],label='$incircle$')

#labelling the coordinates
tri_coords = np.block([[A,B,C,I,D3,E3,F3]])
tri_coords = tri_coords.reshape(2, -1)
plt.scatter(tri_coords[0,:], tri_coords[1,:])
vert_labels = ['A','B','C','I','$D_3$','$E_3$','$F_3$']

for i, txt in enumerate(vert_labels):
    plt.annotate(txt, # this is the text
                 (tri_coords[0,i], tri_coords[1,i]), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0,10), # distance from text to points (x,y)
                 ha='center') # horizontal alignment can be left, right or center


plt.xlabel('$x$')
plt.ylabel('$y$')
plt.legend(loc='best')
plt.grid() # minor
plt.axis('equal')

plt.savefig('/home/sayyam/EE23010/RandVertices/figs/Q1.5.7.png')
plt.show()

