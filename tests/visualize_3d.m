% Test 3D project_and_slide algorithm in bouncing case
% Data can be re-generated in Planktos using a breakpoint in
% _project_and_slide with the visualtest_3d.py file.

triangles_X = [29.277224, 29.277224;
               29.018223, 29.277224;
               29.277224, 27.927223];

triangles_Y = [159.05, 159.05;
               159.95, 159.95;
               159.95, 159.95];
           
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

figure
hold on
fill3(triangles_X, triangles_Y, triangles_Z, 1, 'facealpha', 0.5)
sld1 = plot3(line_segments_x, line_segments_y, line_segments_z,...
             'linewidth', 2);
sld2 = plot3(bnc_segments_x, bnc_segments_y, bnc_segments_z,...
             'linewidth', 2);
orig = quiver3(orig_x(1), orig_y(1), orig_z(1),...
        orig_x(2)-orig_x(1), orig_y(2)-orig_y(1), orig_z(2)-orig_z(1),...
        'linewidth', 2);
legend([orig, sld1, sld2], 'original heading', 'new path', 'solution path')