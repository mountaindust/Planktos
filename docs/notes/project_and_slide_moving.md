Assume a frictionless surface on a 2D mesh element (line segment) parameterized 
by the equation $q(s,t) = \mathbf{Q}_0(t) + s\mathbf{Q}(t)$ for $0\leq s\leq 1$ 
and $t\geq 0$. $\mathbf{Q}_0(t)$ is the first endpoint of the mesh element at 
any time $t$ and $\mathbf{Q}(t) = \mathbf{Q}_1(t) - \mathbf{Q}_0(t)$ is the 
vector from the first endpoint to the second endpoint of the mesh element.

Because the surface is frictionless, movement and deformation of the mesh 
element will not impart any force onto an agent touching the surface except in 
the direction orthogonal to the mesh element. The agent's movement will keep it 
pressed up against the surface until the speed of the element in the direction 
orthogonal to the element is greater than the speed of the agent in that same 
direction, so the movement of the agent orthogonal to the mesh element is the 
same as the mesh element itself. In the direction of the mesh element, movement 
of the agent is due to vector projection of its earlier movement.

Thus the change in agent position is given by
$$\frac{d\mathbf{x}}{dt} = \frac{d\mathbf{x}_{agent\ \parallel\ \mathbf{Q}}}{dt} + 
\frac{d\mathbf{x}_{elem\ \perp\ \mathbf{Q}}}{dt}$$

Let $\mathbf{v}$ be the vector of agent movement before considering contact with 
an immersed boundary. Then
$$\frac{d\mathbf{x}_{agent\ \parallel\ \mathbf{Q}}}{dt} = 
\frac{\mathbf{Q}(t)(\mathbf{Q}(t)\cdot\mathbf{v})}{||\mathbf{Q}(t)||^2}.$$

Consider the change in a position $\mathbf{x}$ along a mesh element due only to 
movement in the mesh element. $\mathbf{x}(t)$ can be parameterized as above via 
$$\mathbf{x}(t) = \mathbf{Q}_0(t) + s(t)\mathbf{Q}(t),$$ 
so
$$\frac{d\mathbf{x}_{elem}}{dt} = \frac{d\mathbf{Q}_0}{dt} 
+ s(t)\frac{d\mathbf{Q}}{dt} + \frac{ds}{dt}\mathbf{Q}(t).$$
$s(t)$ can be found by solving the linearly dependent system of equations above.
To avoid situations where $\mathbf{Q}(t)$ may be zero in one dimension, the 
solution can be written as
$$s(t) = \frac{||\mathbf{x}(t)-\mathbf{Q}_0(t)||}{||\mathbf{Q}(t)||}$$
since $s(t)$ is a non-negative scalar function. One can also leverage the linear 
dependence of the system to avoid that case - in particular, it is worth noting 
that $s(t)$ is therefore linear in $\mathbf{x}$, 
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
rotates so that it begins to move faster than the agent in the direction 
orthogonal to the mesh element or the agent reaches an endpoint of the 
mesh element. We can detect the first case by examining the difference of the 
two speeds in the direction orthogonal to the mesh element. If there is a sign 
change during the time step, the time of separation can be solved for using a 
bracketed root-finding algorithm. We use Brent's method as implemented by scipy.

For the second case detection, the situation is complicated. $\mathbf{x}(t)$ can 
intersect $\mathbf{Q}_0$ or $\mathbf{Q}_1$ at most twice because it can change 
direction once on the mesh element, and also because 
$||\mathbf{Q}_1(t)-\mathbf{Q}_0(t)||^2$ is quadratic and concave up, so it could 
shrink and then grow again. In a really pathological case, $x(t)$ could move off 
of one edge, come back on the mesh element, and then move off the other edge. 
It's probably impossible to move off the same edge twice and then come back on... 
but anyway, we only need to find the first time any of this happens.

Let $\mathbf{Q}(t) = \mathbf{Q}_1(t)-\mathbf{Q}_0(t)$. 
Then $||\mathbf{Q}_1(t)-\mathbf{Q}_0(t)||^2$ has a critical point at
$$t^* = \frac{\mathbf{Q}(0)\cdot(1-\mathbf{Q}(1))}{
    (\mathbf{Q}(0)+\mathbf{Q}(1))\cdot\mathbf{1}-2(\mathbf{Q}(0)\cdot\mathbf{Q}(1))}.$$
If $x(t)$ changes direction, it does so when 
$$\frac{d\mathbf{x}_{agent\ \parallel\ \mathbf{Q}}}{dt}=0 \implies
\mathbf{Q}(t)\cdot\mathbf{v}=0$$
This occurs when
$$t^{**} = \frac{\mathbf{Q}(0)\cdot\mathbf{v}}{(\mathbf{Q}(0)-\mathbf{Q}(1))\cdot\mathbf{v}}$$

If we check both of these critical times (if they exist, and if they are within 
the interval starting at the time of intersection and ending at $t=1$), plus the 
situation at $t=1$, we should have everything covered on detection.

Then, actually finding $t$ is a nonlinear least squares minimization problem 
(this avoids numerical issues versus a root finding algorithm on the difference 
between $x(t)$ and $\mathbf{Q}_0$ or $\mathbf{Q}_1$), and we solve it using 
Powell's hybrid "dog leg" method, a rectangular trust-region method implemented 
in scipy that makes use of the Jacobian of the residual function, here simply 
$\frac{d\mathbf{x}}{dt}-\frac{d\mathbf{Q}}{dt}$.
