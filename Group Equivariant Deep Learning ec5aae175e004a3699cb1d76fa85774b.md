# Group Equivariant Deep Learning

# Lecture 1 - Regular group convolutions

## Lecture 1.1 - Introduce

### motivation: Geometric guarantees(invariance)

Issues:

- Still no guarantee of invariance
- Valuable net capacity is spend on learning invariance
- Redundancy in feature repr

[https://distill.pub/2020/circuits/equivariance](https://distill.pub/2020/circuits/equivariance)

### Geometric guarantees(equivariance)

Normal CNNs are not rotation equivariant

Group equivariant CNNs are rotation euqivariant

Equivariance allows for increased weight sharing

Importance of equivariance:

- No information is lost when the input is transformed
- Guaranteed stability to (local + global) transformations

Group convolutions:

- Equivariance beyond translations
- Geometric guarantees
- Increased weight sharing

G-CNNS are not only relevant for invariant problems but for any type of structured data!

Equivariant problem:

- N-body problem (force/velocity prediction)
- Molecule conformer generation

Invariance problem:

- Molecule property prediction

Symmetries in nature

## Lecture 1.2 - Group Theory | The basics

Introduce what is a group:

### Group Theory Basics

A group  $(G, \cdot)$ is a set of elements $G$  equipped with a group product $\cdot$, a binary operator, that satisfies the following four axioms:

- **Closure**: Given two elements   $g$ and $h$ of $G$, the product  $g \cdot h$  is also in $G$.
- **Associativity**: For$g, h, i \in G$  the product   $\cdot$  is associative, i.e., $g \cdot (h \cdot i) = (g \cdot h) \cdot i$ .
- **Identity element**: There exists an identity element $e \in G$  such that $e \cdot g = g \cdot e = g$   for any $g \in G$
- **Inverse element**: For each  $g \in G$  there exists an inverse element  $g^{-1} \in G$ s.t. $g^{-1} \cdot g = g \cdot g^{-1} = e$.

### Translation group ($\mathbb{R}^2, +$)

The translation group consists of all possible translations in  $\mathbb{R}^2$  and is equipped with the group product and group inverse:

 $g \cdot g' = (x + x')$ 
 $g^{-1} = (-x)$ 

with  $g = (x), g' = (x')$  and  $x, x' \in \mathbb{R}^2$ .

![Untitled](Group%20Equivariant%20Deep%20Learning%20ec5aae175e004a3699cb1d76fa85774b/Untitled.png)

### Roto-translation group  $SE(2)$

The group  $SE(2) = \mathbb{R}^2 \times SO(2)$  consists of the coupled space  $\mathbb{R}^2 \times S^1$  of translations vectors in  $\mathbb{R}^2$ , and rotations in  $SO(2)$  (or equivalently orientations in  $S^1$ ), and is equipped with the group product and group inverse:

 $g \cdot g' = (x, R_\theta) \cdot (x', R_{\theta'}) = (R_\theta x' + x, R_\theta R_{\theta'})$ 
 $g^{-1} = (x, R_\theta)^{-1} = (-R_\theta^{-1}x, R_\theta^{-1})$ 

with  $g = (x, R_\theta), g' = (x', R_{\theta'})$ .

![Untitled](Group%20Equivariant%20Deep%20Learning%20ec5aae175e004a3699cb1d76fa85774b/Untitled%201.png)

### Roto-translation group  $SE(2)$

**Matrix representation**: The group can also be represented by matrices

$g = (x, R_\theta) \leftrightarrow G = \begin{pmatrix}
\cos \theta & -\sin \theta & x \\
\sin \theta & \cos \theta & y \\
0 & 0 & 1
\end{pmatrix} = \left( R_\theta \quad x \atop 0^T \quad 1 \right)$

with the group product and inverse simply given by the matrix product and matrix inverse.

In parametric form:
$(x, \theta) \cdot (x', \theta') = (R_\theta x' + x, \theta + \theta' \mod 2\pi)$

In matrix form:

$\begin{pmatrix}
R_\theta & x \\
0^T & 1
\end{pmatrix}
\cdot
\begin{pmatrix}
R_{\theta'} & x' \\
0^T & 1
\end{pmatrix}=\begin{pmatrix}
R_{\theta + \theta'} & R_\theta x' + x \\
0^T & 1
\end{pmatrix}$

### Scale-translation group  $\mathbb{R}^2 \rtimes \mathbb{R}^+$

The scale-translation group of space  $\mathbb{R}^2 \times \mathbb{R}^+$  of translations vectors in  $\mathbb{R}^2$  and scale/dilation factors in  $\mathbb{R}^+$ , and is equipped with the group product and group inverse:

$g \cdot g' = (x, s) \cdot (x', s') = (sx' + x, ss')$

$g^{-1} = \left( -\frac{1}{s}x, \frac{1}{s} \right)$

with  $g = (x, s) ,  g' = (x', s')$ .

with  $g \cdot g^{-1} = e = (0,1)$ 

matrix representation:$G = \begin{pmatrix}
I_{s} & x \\
0^T & 1
\end{pmatrix}$

![Untitled](Group%20Equivariant%20Deep%20Learning%20ec5aae175e004a3699cb1d76fa85774b/Untitled%202.png)

### Affine groups  $G = \mathbb{R}^d \rtimes H$

Affine groups are semi-direct product groups of some group  $H$  with an action on $\mathbb{R}^d$ from which we derive the following group product and inverse

$g \cdot g' = (x, h) \cdot (x', h') = (h \cdot x' + x, h \cdot h')$

$g^{-1} = (-h^{-1} \cdot x, h^{-1})$

with group elements  $g = (x, h) ,  g' = (x', h')$ .

### Representations

A representation  $\rho : G \rightarrow GL(V)$  is a group homomorphism from G to the general linear group  $GL(V)$ .

That is  $\rho(g)$  is a linear transformation that is parameterized by group elements  $g \in G$  that transforms some vector  $v \in V$  (e.g. an image) such that

$\rho(g') \circ \rho(g)[v] = \rho(g' \cdot g)[v]$

### Left-regular Representations

Example:

- $f \in L^2(\mathbb{R}^2)$  - a 2D image
- $G = SE(2)$  - the roto-translation group
- $\mathcal{L}g(f)(y) = f(R{g^{-1}}(y - x))$  - a roto-translation of the image

A left-regular representation  $\mathcal{L}_g$  is a representation that transforms functions  $f$  by transforming their domains via the inverse group action$\mathcal{L}_g[f](x) := f(g^{-1} \cdot x)$

"Group action" equals group product when domain is  $G$ 

### Group actions

Group product (the action on $G$)
$g \cdot g'$

Left regular representation (the action on  $\mathbb{L}_2(X)$ )
$\mathcal{L}_g f$

Group action (the action on  $\mathbb{R}^d$ )
$g ⨀ x$

### Equivariance

Equivariance is a property of an operator  $\Phi : X \rightarrow Y$  (such as a neural network layer) by which it commutes with the group action:
$\Phi \circ \rho^X(g) = \rho^Y(g) \circ \Phi$

group representation action on $X$

## Lecture 1.3 - Regular group convolutions | Template matching viewpoint

##