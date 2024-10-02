In a given time step, a 2D mesh element (line segment) moves and deforms to a 
different position in space. We use linear interpolation to relate the start 
and end positions of the two points that define this line segment. This creates 
a mulltilinear polynomial in space and time, and we need to check to see if line 
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
it's new position $Q_2$. Then $(x_0,y_0)$ satisfies the equation
$$(x_0,y_0) = Q_0 + \overrightarrow{v}_0 t$$
for some $t\in[0,1]$ with $\overrightarrow{v}_0 = Q_2 - Q_0$. Similarly, let 
$(x_1,y_1)$ be on the line segement given by linear interpolatino of $Q_1$ to 
it's new position $Q_3$. Then
$$(x_1,y_1) = Q_1 + \overrightarrow{v}_1 t$$
with $\overrightarrow{v}_1 = Q_3 - Q_1$.

The equation for a line through $(x_0(t),y_0(t))$ and $(x_1(t),y_1(t))$ is
$$(y_0-y_1)x + (x_1-x_0)y + (x_0y_1-x_1y_0) = 0.$$

We now replace $x$ and $y$ with the respective components of the vector 
$\overrightarrow{x} = P_0 + \overrightarrow{w}t$ from the equation of the 
agent's position in order to find intersections. Since $x_0,x_1,y_0,y_1$ are 
linear functions of $t$, and $x$ and $y$ are linear functions of $t$, the result 
is an equation for the roots of a quadratic equation in $t$:
$$\begin{gather*}
At^2 + Bt + C = 0\\
A = (v_{0,0}v_{1,1}-v_{1,0}v_{0,1}) + (v_{0,1}-v_{1,1})w_0 + (v_{1,0}-v_{0,0})w_1\\
B = (Q_{0,0}v_{1,1}+Q_{1,1}v_{0,1}-Q_{0,1}v_{1,0}) + 
(Q_{0,1}-Q_{1,1})w_0 + (v_{0,1}-v_{1,1})P_{0,0} +
(Q_{1,0}-Q_{0,0})w_1 + (v_{1,0}-v_{0,0})P_{0,1}\\
C = (Q_{0,0}Q_{1,1}-Q_{1,0}Q_{0,1}) + (Q_{0,1}-Q_{1,1})P_{0,0} + (Q_{1,0}-Q_{0,0})P_{0,1}
\end{gather*}$$
where $Q_0 = (Q_{0,0},Q_{0,1})$, $P_0=(P_{0,0},P_{0,1})$, 
$\overrightarrow{v}_0=(v_{0,0},v_{0,1})$, etc.

If solutions to this quadatic lie within $[0,1]$, then the agent intersects the 
interpolated (infinite) line through the mesh vertices. We then have to check 
that the intersection is between the vertices $(x_0(t),y_0(t))$ and 
$(x_1(t),y_1(t))$. If two solutions satisfy all of this, we take the smaller one 
since it will be the first interesection in time (two intersections can be 
achieved by rotation of the mesh element).

Note that in some cases, partiuclarly if the mesh element does not actually 
move, the quadratic is actually a linear equation. So we must test for this in 
order to avoid dividing by zero. In other cases, the quadratic may have no real 
solutions. This happens when the line formed by the mesh vertices rotates around 
the agent as it moves past.