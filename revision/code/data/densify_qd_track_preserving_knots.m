function [track, target_distance_m, diagnostic] = ...
    densify_qd_track_preserving_knots(track, samples_per_meter, planned_distance_m)
%DENSIFY_QD_TRACK_PRESERVING_KNOTS Subdivide long edges without dropping input knots.
%   The input positions define the target-time polyline. Every source position is copied
%   bit-for-bit into the dense track; extra positions are inserted only within a source
%   edge. This avoids qd_track.interpolate_positions, whose equal-distance resampling
%   may discard target-time vertices and shorten a curved track.

if nargin ~= 3 || ~isa(track, 'qd_track')
    error('T08:DensifyInput', 'Expected a qd_track, samples_per_meter and planned distances.');
end
if ~(isnumeric(samples_per_meter) && isreal(samples_per_meter) && ...
        isscalar(samples_per_meter) && isfinite(samples_per_meter) && samples_per_meter > 0)
    error('T08:DensifyDensity', 'samples_per_meter must be a finite positive scalar.');
end

source_positions = double(track.positions);
source_orientation = double(track.orientation);
source_initial_position = double(track.initial_position);
source_segment_index = double(track.segment_index(:).');
source_scenario = track.scenario;
planned_distance_m = double(planned_distance_m(:).');
n_source = size(source_positions, 2);
if size(source_positions,1) ~= 3 || n_source < 2 || ...
        numel(planned_distance_m) ~= n_source || ...
        any(~isfinite(source_positions(:))) || any(~isfinite(planned_distance_m)) || ...
        any(diff(planned_distance_m) <= 0) || ...
        any(source_segment_index < 1) || any(source_segment_index > n_source)
    error('T08:DensifyGeometry', 'Track and target distances must be finite and strictly increasing.');
end
if ~isempty(source_orientation) && ...
        (~isequal(size(source_orientation), [3,n_source]) || any(~isfinite(source_orientation(:))))
    error('T08:DensifyOrientation', 'Track orientation must be empty or finite [3,N].');
end

source_edge = diff(source_positions,1,2);
source_gap_m = sqrt(sum(abs(source_edge).^2,1));
if any(source_gap_m <= 0)
    error('T08:ZeroLengthEdge', 'Repeated source positions are not allowed in a moving track.');
end
max_source_gap_m = max(source_gap_m);
max_gap_m = 1 / double(samples_per_meter);
n_subdivision = max(1, ceil(source_gap_m ./ max_gap_m));
n_dense = 1 + sum(n_subdivision);
dense_positions = zeros(3,n_dense);
target_knot_index = zeros(1,n_source);
dense_positions(:,1) = source_positions(:,1);
target_knot_index(1) = 1;
cursor = 1;
for i_edge = 1:(n_source-1)
    n_sub = n_subdivision(i_edge);
    delta = source_positions(:,i_edge+1) - source_positions(:,i_edge);
    for i_sub = 1:n_sub
        cursor = cursor + 1;
        if i_sub == n_sub
            dense_positions(:,cursor) = source_positions(:,i_edge+1);
        else
            dense_positions(:,cursor) = source_positions(:,i_edge) + ...
                delta .* (double(i_sub) / double(n_sub));
        end
    end
    target_knot_index(i_edge+1) = cursor;
end

dense_edge = diff(dense_positions,1,2);
dense_gap_m = sqrt(sum(abs(dense_edge).^2,1));
max_dense_gap_m = max(dense_gap_m);
gap_roundoff_bound_m = 64 * eps(max([1,max_gap_m,max_dense_gap_m]));
if max_dense_gap_m > max_gap_m + gap_roundoff_bound_m
    error('T08:DenseGap', 'Densified track exceeds its maximum spatial gap.');
end

source_distance_m = [0,cumsum(source_gap_m)];
dense_distance_m = [0,cumsum(dense_gap_m)];
target_distance_m = dense_distance_m(target_knot_index);
source_absolute_positions = source_positions + ...
    source_initial_position(:,ones(1,n_source));
source_coordinate_scale_m = max([1; abs(source_initial_position(:)); ...
    abs(source_absolute_positions(:))]);
path_scale_m = max([1,source_distance_m(end),dense_distance_m(end)]);
roundoff_scale_m = max(source_coordinate_scale_m,path_scale_m);
distance_roundoff_bound_m = ...
    (8 * max(1,n_source-1) + 64) * eps(roundoff_scale_m);
max_plan_distance_roundoff_m = max(abs(target_distance_m - planned_distance_m));
if max_plan_distance_roundoff_m > distance_roundoff_bound_m
    error('T08:PlanDistanceMismatch', ...
        'Planned target chord distances differ from measured track geometry beyond roundoff.');
end

original_knot_positions = dense_positions(:,target_knot_index);
max_original_target_position_error_m = max(vecnorm( ...
    original_knot_positions - source_positions, 2, 1));
if max_original_target_position_error_m > 0
    error('T08:KnotPositionChanged', 'At least one original target position was altered.');
end

if ~isempty(source_orientation)
    dense_orientation = zeros(3,n_dense);
    dense_orientation(1,:) = qf.slerp( ...
        source_distance_m, source_orientation(1,:), 0, dense_distance_m, true);
    [dense_orientation(3,:),dense_orientation(2,:)] = qf.slerp( ...
        source_distance_m, source_orientation(3,:), source_orientation(2,:), ...
        dense_distance_m, true);
    dense_orientation(:,target_knot_index) = source_orientation;
else
    dense_orientation = [];
end

track.positions = dense_positions;
track.orientation = dense_orientation;
track.segment_index = target_knot_index(source_segment_index);
track.scenario = source_scenario;
[measured_length_m, measured_distance_m] = track.get_length();
if abs(measured_length_m - dense_distance_m(end)) > ...
        64 * eps(max([1,measured_length_m,dense_distance_m(end)]))
    error('T08:TrackLengthMismatch', 'qd_track.get_length disagrees with densified geometry.');
end
target_distance_m = measured_distance_m(target_knot_index);

diagnostic = struct();
diagnostic.target_snapshot_count = n_source;
diagnostic.dense_snapshot_count = n_dense;
diagnostic.target_knot_index = target_knot_index;
diagnostic.target_knot_distance_m = target_distance_m;
diagnostic.max_source_gap_m = max_source_gap_m;
diagnostic.max_dense_gap_m = max_dense_gap_m;
diagnostic.spatial_gap_roundoff_bound_m = gap_roundoff_bound_m;
diagnostic.max_original_target_position_error_m = max_original_target_position_error_m;
diagnostic.max_plan_distance_roundoff_m = max_plan_distance_roundoff_m;
diagnostic.distance_roundoff_bound_m = distance_roundoff_bound_m;
diagnostic.movement_profile_roundoff_bound_m = distance_roundoff_bound_m;
diagnostic.movement_profile_endpoint_clamp_bound_m = distance_roundoff_bound_m;
diagnostic.movement_profile_endpoint_clamp_applied = false;
diagnostic.measured_target_chord_length_m = target_distance_m(end);
diagnostic.measured_dense_track_length_m = measured_length_m;
diagnostic.scenario_segment_index = double(track.segment_index(:).');
end
