% Test 3D project_and_slide algorithm in bouncing case
% Data can be re-generated in Planktos using a breakpoint in
% _project_and_slide with the visualtest_3d.py file.

triangles_X = [29.277224, 29.277224;
               29.018223, 29.277224;
               29.277224, 27.927223];

triangles_Y = [159.45, 159.45;
               159.75, 159.75;
               159.75, 159.75];
           
triangles_Z = [56.443504, 56.443504;
               56.9905,   56.443504;
               56.443504, 56.648506];
           
line_segments_x = [29.065876, 29.277223, 29.435617];
line_segments_y = [159.603829, 159.617881, 159.661421];
line_segments_z = [56.475598, 56.443504, 56.108984];

% old
% bnc_segments_x = [29.277223, 29.277223];
% bnc_segments_y = [159.617881, 159.990558];
% bnc_segments_z = [56.443504, 56.443504];
% new
bnc_segments_x = [29.277223, 29.277223];
bnc_segments_y = [159.617881, 159.661421];
bnc_segments_z = [56.443504, 56.443504];

orig_x = [29.065876, 29.833577];
orig_y = [159.603829, 159.661421];
orig_z = [56.475598, 55.695334];

f = figure('Position',[100,100,1680,1260]);
hold on
fill3(triangles_X, triangles_Y, triangles_Z, 1, 'facealpha', 0.5)
stpt = plot3(orig_x(1), orig_y(1), orig_z(1), 'k.','MarkerSize',32);
sld1pt = quiver3(line_segments_x(2), line_segments_y(2), line_segments_z(2),...
             line_segments_x(3)-line_segments_x(2),...
             line_segments_y(3)-line_segments_y(2),...
             line_segments_z(3)-line_segments_z(2),...
             'linewidth', 6);
sld2 = plot3(bnc_segments_x, bnc_segments_y, bnc_segments_z,...
             'linewidth', 6);
orig = quiver3(orig_x(1), orig_y(1), orig_z(1),...
        orig_x(2)-orig_x(1), orig_y(2)-orig_y(1), orig_z(2)-orig_z(1),...
        'linewidth', 6);
sld1_clr = get(sld1pt,'Color');
stop_pt = plot3(bnc_segments_x(2), bnc_segments_y(2), bnc_segments_z(2),...
                'kp','MarkerSize',32,'MarkerFaceColor','k');
sld1 = plot3(line_segments_x(1:2), line_segments_y(1:2), line_segments_z(1:2),...
             'linewidth', 6, 'Color', sld1_clr);
leg = legend([stpt, orig, sld1pt, sld2, stop_pt], 'start point', 'original heading',...
       'new path', 'solution path', 'end point');
set(gca,'fontsize',32)
xlabel('x')
ylabel('y')
zlabel('z')
campos([12.6553  160.2790   61.0441])
set(leg, 'Position', [0.6, 0.2713, 0.23, 0.1952])
exportgraphics(gca, 'proj_3d.pdf', 'ContentType', 'vector')