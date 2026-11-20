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
$\mathbf{x}(t) = \mathbf{Q}_0(t) + s(t)\mathbf{Q}(t),$ 
so
$$\frac{d\mathbf{x}_{elem}}{dt} = \frac{d\mathbf{Q}_0}{dt} 
+ s(t)\frac{d\mathbf{Q}}{dt} + \frac{ds}{dt}\mathbf{Q}(t).$$
$s(t)$ is the signed coordinate of $\mathbf{x}$ along the element, found by 
projecting $\mathbf{x}-\mathbf{Q}_0$ onto $\mathbf{Q}$:
$$s(t) = \frac{(\mathbf{x}(t)-\mathbf{Q}_0(t))\cdot\mathbf{Q}(t)}{||\mathbf{Q}(t)||^2}.$$
This is defined whatever the orientation of the element and is linear in 
$\mathbf{x}$. It is 0 at $\mathbf{Q}_0$ and 1 at $\mathbf{Q}_1$, negative past 
$\mathbf{Q}_0$ and greater than 1 past $\mathbf{Q}_1$. The sign matters because 
the path is also evaluated past the ends of the element, when finding where the 
agent leaves it. Note that the $ds/dt$ term will disappear when we project onto 
a direction perpendicular to $\mathbf{Q}(t)$.

Now let $\mathbf{Q}_\perp(t)$ be a vector orthogonal to $\mathbf{Q}(t)$. Then
$$\frac{d\mathbf{x}_{elem\ \perp\ \mathbf{Q}}}{dt} =
\frac{\mathbf{Q}_\perp(t)\left[\mathbf{Q}_\perp(t)\cdot\left(
    \frac{d\mathbf{Q}_0}{dt}+s(t)\frac{d\mathbf{Q}}{dt}
    \right)\right]}{||\mathbf{Q}_\perp(t)||^2},$$
so
$$\frac{d\mathbf{x}}{dt} = 
\frac{\mathbf{Q}(t)(\mathbf{Q}(t)\cdot\mathbf{v})}{||\mathbf{Q}(t)||^2} +
\frac{\mathbf{Q}_\perp(t)\left[\mathbf{Q}_\perp(t)\cdot\left(
    \frac{d\mathbf{Q}_0}{dt}+s(t)\frac{d\mathbf{Q}}{dt}
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

For the second case, measure the agent's place along the element by its signed 
distance from $\mathbf{Q}_0$. Multiplying the definition of $s(t)$ by 
$||\mathbf{Q}(t)||$ gives
$$\sigma(t) = s(t)||\mathbf{Q}(t)|| = 
\frac{(\mathbf{x}(t)-\mathbf{Q}_0(t))\cdot\mathbf{Q}(t)}{||\mathbf{Q}(t)||} = 
(\mathbf{x}(t)-\mathbf{Q}_0(t))\cdot\frac{\mathbf{Q}(t)}{||\mathbf{Q}(t)||},$$
the scalar projection of $\mathbf{x}-\mathbf{Q}_0$ onto the unit vector along the 
element. Where $s$ measures position as a fraction of the element's length, 
$\sigma$ measures it in units of length. Measure the agent's place relative to 
the other end by its signed distance past $\mathbf{Q}_1$, 
$h(t) = \sigma(t)-||\mathbf{Q}(t)||$. 
The agent is off the element when $\sigma<0$ or $h>0$. Projecting the equation 
for $d\mathbf{x}/dt$ onto $\mathbf{Q}$, and using that $\mathbf{x}-\mathbf{Q}_0$ 
is parallel to $\mathbf{Q}$ (so the derivative of $\mathbf{Q}/||\mathbf{Q}||$ 
does not contribute), gives
$$\frac{d\sigma}{dt} = \frac{\mathbf{Q}(t)\cdot\left(\mathbf{v}-
    \frac{d\mathbf{Q}_0}{dt}\right)}{||\mathbf{Q}(t)||}, \qquad
\frac{dh}{dt} = \frac{\mathbf{Q}(t)\cdot\left(\mathbf{v}-
    \frac{d\mathbf{Q}_1}{dt}\right)}{||\mathbf{Q}(t)||}.$$
Along the element, then, the agent moves relative to the endpoint in question, 
not with its own velocity $\mathbf{v}$.

The agent meets the element at the time of intersection $t_I$, so detection 
concerns $t\in[t_I,1]$. Let $u = (t-t_I)/(1-t_I)$, which runs from 0 to 1 over 
that interval. The endpoints of the element move linearly in time, so 
$d\mathbf{Q}_0/dt$ and $d\mathbf{Q}_1/dt$ are constant and $\mathbf{Q}$ is the 
linear interpolation
$$\mathbf{Q}(u) = (1-u)\mathbf{Q}_{t_I} + u\mathbf{Q}_{end},$$
where $\mathbf{Q}_{t_I}$ and $\mathbf{Q}_{end}$ are the element vectors at 
$t=t_I$ and $t=1$. Each numerator above therefore has the form 
$\mathbf{Q}(u)\cdot\mathbf{w}$, where $\mathbf{w}$ is 
$\mathbf{v}-d\mathbf{Q}_0/dt$ for $\sigma$ and $\mathbf{v}-d\mathbf{Q}_1/dt$ for 
$h$. Both are constant over the step, since $\mathbf{v}$ is the agent's movement 
for the step and the endpoint velocities are constant. Expanding,
$$\mathbf{Q}(u)\cdot\mathbf{w} = (1-u)\,\mathbf{Q}_{t_I}\cdot\mathbf{w} + 
u\,\mathbf{Q}_{end}\cdot\mathbf{w} = \mathbf{Q}_{t_I}\cdot\mathbf{w} - 
u\,(\mathbf{Q}_{t_I}-\mathbf{Q}_{end})\cdot\mathbf{w},$$
which is linear in $u$. If $(\mathbf{Q}_{t_I}-\mathbf{Q}_{end})\cdot\mathbf{w} = 0$ 
it is constant and does not change sign. Otherwise it vanishes exactly once, at
$$u^* = \frac{\mathbf{Q}_{t_I}\cdot\mathbf{w}}
    {(\mathbf{Q}_{t_I}-\mathbf{Q}_{end})\cdot\mathbf{w}},$$
and changes sign there. Only a root with $0<u^*<1$ lies within the step; it is 
mapped back to time by $t = t_I + u^*(1-t_I)$.

Since $||\mathbf{Q}||>0$, each derivative has the sign of its numerator, so it 
changes sign at most once on $[t_I,1]$, at its own $u^*$. $\sigma$ and $h$ are 
therefore each unimodal on $[t_I,1]$: the minimum of $\sigma$ and the maximum of 
$h$ lie at $t_I$, at $t=1$, or at their own $u^*$. The agent starts on the 
element, so sampling $\sigma$ and $h$ at $t=1$ and at those critical times detects every excursion past either end, 
including one that begins and ends within the step.

Unimodality also makes the set of times past a given end a single interval. 
When a sample $t_s$ is past an end, the time the agent leaves that end is the 
only root of $\sigma$ (or $h$) between $t_s$ and the last time before it at which 
the agent is on the element: $t_I$, or the critical time when $\sigma$ (or $h$) 
first moves away from that end to an extreme there. That bracket is solved with 
Brent's method. If the agent goes past both ends during the step, it leaves at 
whichever it reaches first. Each evaluation of $\sigma$ integrates 
$d\mathbf{x}/dt$ from $t_I$, so $\sigma$ is computed from the same path as the 
agent's position.
