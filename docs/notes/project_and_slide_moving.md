Assume a frictionless surface on a 2D mesh element (line segment) parameterized 
by the equation $q(s,t) = \mathbf{Q}_0(t) + s\mathbf{Q}(t)$ for $0\leq s\leq 1$ 
and $t\geq 0$. $\mathbf{Q}_0(t)$ is the first endpoint of the mesh element at 
any time $t$ and $\mathbf{Q}(t) = \mathbf{Q}_1(t) - \mathbf{Q}_0(t)$ is the 
vector from the first endpoint to the second endpoint of the mesh element.

Because the surface is frictionless, movement and deformation of the mesh 
element will not impart any force onto an agent touching the surface except in 
the direction orthogonal to the mesh element. It is assumed that the agent's 
movement will keep it pressed up against the surface during the time step, so 
all movement of the agent orthogonal to the surface can be explained by the 
movement of the mesh element. In the direction of the mesh element, movement of 
the agent is due to vector projection of its earlier movement.

Thus the change in agent position is given by
$$\frac{d\mathbf{x}}{dt} = \frac{d\mathbf{x}_{agent\ \parallel\ \mathbf{Q}}}{dt} + 
\frac{d\mathbf{x}_{elem\ \perp\ \mathbf{Q}}}{dt}$$

Let $\mathbf{v}$ be the vector of agent movement before considering contact with 
an immersed boundary. Then
$$\frac{d\mathbf{x}_{agent\ \parallel\ \mathbf{Q}}}{dt} = 
\frac{\mathbf{Q}(t)(\mathbf{Q}(t)\cdot\mathbf{v})}{||\mathbf{Q}(t)||^2}.$$

Consider the change in a position $\mathbf{x}$ along a mesh element due only to 
movement in the mesh element. $\mathbf{x}(t)$ can be parameterized as above via 
$\mathbf{x}(t) = \mathbf{Q}_0(t) + s(t)\mathbf{Q}(t)$, so
$$\frac{d\mathbf{x}_{elem}}{dt} = \frac{d\mathbf{Q}_0}{dt} 
+ s(t)\frac{d\mathbf{Q}}{dt} + \frac{ds}{dt}\mathbf{Q}(t).$$
$s(t)$ can be found by solving the system of $\mathbf{x}$ equations
$$s(t) = \frac{||\mathbf{x}(t)-\mathbf{Q}_0(t)||}{||\mathbf{Q}(t)||}.$$
It is also possible to formulate this in a way that is linear in $\mathbf{x}$, 
but it is more verbose in notation. Note that the $ds/dt$ term will disappear 
when we project onto a direction perpendicular to $\mathbf{Q}(t)$.

Now let $\mathbf{Q}_\perp(t)$ be a vector orthogonal to $\mathbf{Q}(t)$. Then
$$\frac{d\mathbf{x}_{elem\ \perp\ \mathbf{Q}}}{dt} =
\frac{\mathbf{Q}_\perp(t)\left[\mathbf{Q}_\perp(t)\cdot\left(
    \frac{d\mathbf{Q}_0}{dt}+\frac{d\mathbf{Q}}{dt}
    \frac{||\mathbf{x}(t)-\mathbf{Q}_0(t)||}{||\mathbf{Q}(t)||}
    \right)\right]}{||\mathbf{Q}_\perp(t)||^2},$$
so
$$\frac{d\mathbf{x}}{dt} = 
\frac{\mathbf{Q}(t)(\mathbf{Q}(t)\cdot\mathbf{v})}{||\mathbf{Q}(t)||^2} +
\frac{\mathbf{Q}_\perp(t)\left[\mathbf{Q}_\perp(t)\cdot\left(
    \frac{d\mathbf{Q}_0}{dt}+\frac{d\mathbf{Q}}{dt}
    \frac{||\mathbf{x}(t)-\mathbf{Q}_0(t)||}{||\mathbf{Q}(t)||}
    \right)\right]}{||\mathbf{Q}_\perp(t)||^2}.$$
This is reducible to a nonautonomous, linear equation in $\mathbf{x}$ with 
continuous coefficients. A unique solution exists. Rather than formulate the 
solution as an integral equation, we choose to solve this using RK45.

Agent movement can leave the mesh element in two ways. Either the mesh element 
rotates so that it is parallel to the original movement of the agent (and so the 
agent's movement orthogonal to the mesh element transitions from moving toward 
the mesh element to moving away from it) or the agent reaches an endpoint of the 
mesh element. We can detect the first case by looking for times at which the 
dot product $\mathbf{v}_\perp\cdot\mathbf{Q}(t)$ is zero. For the second case, 
we want to find the time at which either $||\mathbf{x}(t)-\mathbf{Q}_0(t)||$ or 
$||\mathbf{x}(t)-\mathbf{Q}_1(t)||=0$. This is a nonlinear least squares 
minimization problem in $t$, and we solve it using Powell's hybrid "dog leg" 
method, a rectangular trust-region method implemented in scipy that makes use of 
the Jacobian of the residual function, here simply 
$\frac{d\mathbf{x}}{dt}-\frac{d\mathbf{Q}}{dt}$.
