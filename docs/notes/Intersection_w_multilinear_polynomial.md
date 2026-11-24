In a given time step, a 2D mesh element (line segment) moves and deforms to a 
different position in space. We use linear interpolation to relate the start 
and end positions of the two points that define this line segment. This creates 
a multilinear polynomial in space and time, and we need to check to see if line 
segments created by agent movement intersect with these.

Let $t\in[0,1]$ parameterize time within the time step (e.g., we normalize the 
time step to be a unit interval). Then the position of the agent during the time 
step can be written as
$$\overrightarrow{x} = P_0 + \overrightarrow{w}t$$
where $P_0$ is the starting position of the agent at $t=0$, 
$\overrightarrow{w}=P_1-P_0$ where $P_1$ is the ending position of the agent at $t=1$,
and $t$ is constrained to $t\in[0,1]$.

We will also parameterize the mesh element in time. Let $Q_0$ and $Q_1$ be the 
points defining the mesh element at $t=0$ and let $Q_2$ and $Q_3$ be the 
position of the same points at time $t=1$.

Let $(x_0,y_0)$ be on the line segment given by linear interpolation of $Q_0$ to 
its new position $Q_2$. Then $(x_0,y_0)$ satisfies the equation
$$(x_0,y_0) = Q_0 + \overrightarrow{v}_0 t$$
for some $t\in[0,1]$ with $\overrightarrow{v}_0 = Q_2 - Q_0$. Similarly, let 
$(x_1,y_1)$ be on the line segment given by linear interpolation of $Q_1$ to 
its new position $Q_3$. Then
$$(x_1,y_1) = Q_1 + \overrightarrow{v}_1 t$$
with $\overrightarrow{v}_1 = Q_3 - Q_1$.

The agent is on this line at time $t$ when the vector from $(x_0,y_0)$ to the
agent is parallel to the vector from $(x_0,y_0)$ to $(x_1,y_1)$, i.e. when their
2D cross product $u\times v = u_0v_1-u_1v_0$ is zero:
$$\big((x_1,y_1)-(x_0,y_0)\big)\times\big(\overrightarrow{x}-(x_0,y_0)\big) = 0.$$

All positions are taken relative to the agent's starting point $P_0$. Let
$$\overrightarrow{a} = Q_0-P_0,\quad \overrightarrow{b} = Q_1-P_0,\quad
\overrightarrow{D} = \overrightarrow{b}-\overrightarrow{a},\quad
\overrightarrow{E} = \overrightarrow{v}_1-\overrightarrow{v}_0,\quad
\overrightarrow{F} = \overrightarrow{w}-\overrightarrow{v}_0.$$
Then $(x_1,y_1)-(x_0,y_0) = \overrightarrow{D}+\overrightarrow{E}t$ is the
element vector, and
$\overrightarrow{x}-(x_0,y_0) = \overrightarrow{F}t-\overrightarrow{a}$. Since
both are linear in $t$, their cross product is a quadratic in $t$:
$$\begin{gather*}
At^2 + Bt + C = 0\\
A = \overrightarrow{E}\times\overrightarrow{F}\\
B = \overrightarrow{D}\times\overrightarrow{F} - \overrightarrow{E}\times\overrightarrow{a}\\
C = \overrightarrow{a}\times\overrightarrow{b}
\end{gather*}$$
using $\overrightarrow{D}\times\overrightarrow{a} = \overrightarrow{b}\times\overrightarrow{a}$
for $C$.

Working relative to $P_0$ matters numerically. When the agent starts within
roundoff of the element's line, the true $C$ is near zero. Here it is the cross
product of the two short vectors from the agent to the element's vertices, and so
is accurate. Formed in absolute coordinates, the same $C$ is a sum of products of
size $|P_0|^2$, and roundoff then decides which side of $t=0$ the root falls on.

The roots are computed as
$$q = -\tfrac{1}{2}\left(B + \operatorname{sign}(B)\sqrt{B^2-4AC}\right),\qquad
t = \frac{q}{A},\quad t = \frac{C}{q},$$
which never subtracts nearly equal numbers. The textbook
$(-B\pm\sqrt{B^2-4AC})/2A$ loses the smaller root when $|4AC|\ll B^2$, which is
the case for a contact near $t=0$.

If solutions to this quadratic lie within $[0,1]$, then the agent intersects the 
interpolated (infinite) line through the mesh vertices. We then have to check 
that the intersection is between the vertices $(x_0(t),y_0(t))$ and 
$(x_1(t),y_1(t))$. If two solutions satisfy all of this, we take the smaller one 
since it will be the first intersection in time (two intersections can be 
achieved by rotation of the mesh element).

Note that in some cases, particularly if the mesh element does not actually 
move, the quadratic is actually a linear equation. So we must test for this in 
order to avoid dividing by zero. In other cases, the quadratic may have no real 
solutions. This happens when the line formed by the mesh vertices rotates around 
the agent as it moves past.