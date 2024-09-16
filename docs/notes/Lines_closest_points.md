Assume we have two skew lines in 3D. Line $A$ is formed by the points 
$(\mathbf{a_0},\mathbf{a_1})$ and Line $B$ via $(\mathbf{b_0},\mathbf{b_1})$.

Let $\mathbf{\hat{{v}}}_1$ be a unit vector in the direction of Line $A$, e.g. 
$\mathbf{\hat{{v}}}_1 = (\mathbf{a_1}-\mathbf{a_2})/||(\mathbf{a_1}-\mathbf{a_2})||$ 
and similarly for $\mathbf{\hat{{v}}}_2$ for Line $B$.

Then $\mathbf{z} = \mathbf{\hat{{v}}}_1\times\mathbf{\hat{{v}}}_2$ is a vector 
that is perpendicular to both lines. It is not a unit vector, but has magnitude 
$\sin(\theta)$ where $\theta$ is the angle between the two skew lines. We assume 
$\theta\neq 0$. 

To get the closest point on Line $A$ to Line $B$, we will consider the plane 
formed by the translation of Line $B$ along the direction of $\mathbf{z}$. 
Wherever Line $A$ intersects this plane is the closest point.

To get the equation of the $B$-translation plane, we note that $\mathbf{b_0}$ is 
a point in the plane and then look for a normal to the plane. A normal is given 
by
$$\mathbf{\hat{{v}}}_2 \times \frac{\mathbf{z}}{||\mathbf{z}||} = 
\frac{(\mathbf{\hat{{v}}}_2 \times \mathbf{z})}{||\mathbf{z}||} \equiv 
\mathbf{\overrightarrow{n}}_B$$

Note that $\mathbf{\overrightarrow{n}}_B$ is a unit vector because 
$\mathbf{\hat{{v}}}_2$ is perpendicular to $\mathbf{z}/||\mathbf{z}||$ and they 
are both unit vectors.

So, an equation for the plane is 
$(\mathbf{x} - \mathbf{b_0})\cdot\mathbf{\overrightarrow{n}}_B = 0$ where 
$\mathbf{x}$ is a variable. The equation for Line $A$ is 
$\mathbf{x} = \mathbf{a_0} + t_0\mathbf{\hat{{v}}}_1$, where $t_0$ is a scalar 
variable.

Setting these two equations equal to each other, we can solve for $t_0$:
$$\begin{gather*}
(\mathbf{a_0}+t_0\mathbf{\hat{{v}}}_1-\mathbf{b_0})\cdot\mathbf{\overrightarrow{n}}_B = 0\\
(\mathbf{a_0}-\mathbf{b_0})\cdot\mathbf{\overrightarrow{n}}_B + 
(\mathbf{\hat{{v}}}_1\cdot\mathbf{\overrightarrow{n}}_B)t_0 = 0\\
t_0 = \frac{(\mathbf{b_0}-\mathbf{a_0})\cdot\mathbf{\overrightarrow{n}}_B}{\mathbf{\hat{{v}}}_1\cdot\mathbf{\overrightarrow{n}}_B}
\end{gather*}$$

Note that $\mathbf{\hat{{v}}}_1\cdot\mathbf{\overrightarrow{n}}_B$ is the triple 
product $\mathbf{\hat{{v}}}_1\cdot(\mathbf{\hat{{v}}}_2 \times \mathbf{z})/||\mathbf{z}||$. 
Since $\mathbf{\hat{{v}}}_1\cdot(\mathbf{\hat{{v}}}_2 \times \mathbf{z}) = 
\mathbf{z}\cdot(\mathbf{\hat{{v}}}_1\times\mathbf{\hat{{v}}}_2) = 
\mathbf{z}\cdot\mathbf{z} = ||\mathbf{z}||^2$, we now have that
$\mathbf{\hat{{v}}}_1\cdot\mathbf{\overrightarrow{n}}_B = ||\mathbf{z}||$. Thus,
$$
t_0 = \frac{(\mathbf{b_0}-\mathbf{a_0})\cdot\mathbf{\overrightarrow{n}}_B}{\mathbf{\hat{{v}}}_1\cdot\mathbf{\overrightarrow{n}}_B} = 
\frac{(\mathbf{b_0}-\mathbf{a_0})\cdot(\mathbf{\hat{{v}}}_2 \times \mathbf{z})}{||\mathbf{z}||^2}.
$$

Finally, we note that the determinant of a matrix $[\mathbf{a}; \mathbf{b}; \mathbf{c}]$, 
with vectors $\mathbf{a}, \mathbf{b}$, and $\mathbf{c}$ slotted in as rows can be 
calculated as $\mathbf{a}\cdot(\mathbf{b}\times\mathbf{c})$. 

This is the basis of the algorithm to find the closest point on Line $A$ to 
Line $B$. To find the closest point on Line $B$ to Line $A$, it is similar:
$$
t_1 = \frac{(\mathbf{a_0}-\mathbf{b_0})\cdot\mathbf{\overrightarrow{n}}_A}{\mathbf{\hat{{v}}}_2\cdot\mathbf{\overrightarrow{n}}_A} = 
\frac{(\mathbf{a_0}-\mathbf{b_0})\cdot(\mathbf{\hat{{v}}}_1 \times \mathbf{z})}{-||\mathbf{z}||^2}
= \frac{(\mathbf{b_0}-\mathbf{a_0})\cdot(\mathbf{\hat{{v}}}_1 \times \mathbf{z})}{||\mathbf{z}||^2},
$$
since the cross product is anticommutative.
