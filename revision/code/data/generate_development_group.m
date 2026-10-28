function generate_development_group(job_json_path, execute_channels)
%GENERATE_DEVELOPMENT_GROUP Explicitly gated one-group development CSI adapter.
% This function writes only one MATLAB-v7 file under revision/data/development.
% Calling it runs real QuaDRiGa channel generation; T08 must not call it.

if nargin < 2 || ~isequal(execute_channels, true)
    error('T08:ExplicitExecutionRequired', ...
        'Real channel generation requires generate_development_group(job,true).');
end
if ~(ischar(job_json_path) || (isstring(job_json_path) && isscalar(job_json_path)))
    error('T08:InvalidJobPath', 'job_json_path must be a scalar path string.');
end
job_json_path = char(job_json_path);
if ~isfile(job_json_path)
    error('T08:MissingJob', 'The per-group JSON job file does not exist.');
end
job = jsondecode(fileread(job_json_path));

this_dir = canonical_path(fileparts(mfilename('fullpath')));
if ~isfield(job, 'project_root') || ...
        ~(ischar(job.project_root) || (isstring(job.project_root) && isscalar(job.project_root)))
    error('T08:ProjectRootMismatch', 'The job must name its real project_root.');
end
project_root = canonical_path(job.project_root);
if ~isfolder(project_root)
    error('T08:ProjectRootMismatch', 'The job project_root must be an existing directory.');
end
live_source_dir = canonical_path(fullfile(project_root, 'revision', 'code', 'data'));
if strcmpi(this_dir, live_source_dir)
    repo_root = project_root;
else
    if ~isfield(job, 'config_hash') || ...
            ~(ischar(job.config_hash) || (isstring(job.config_hash) && isscalar(job.config_hash)))
        error('T08:ProjectRootMismatch', ...
            'A non-live generator path requires a fingerprint-bound source snapshot.');
    end
    config_hash = lower(char(job.config_hash));
    if isempty(regexp(config_hash, '^[0-9a-f]{64}$', 'once'))
        error('T08:ProjectRootMismatch', 'The source snapshot config_hash must be 64 lowercase hex digits.');
    end
    snapshot_root = canonical_path(fullfile(project_root, 'revision', 'data', ...
        'development', 'source_snapshots', config_hash));
    snapshot_source_dir = canonical_path(fullfile(snapshot_root, 'revision', 'code', 'data'));
    if ~strcmpi(this_dir, snapshot_source_dir)
        error('T08:ProjectRootMismatch', ...
            'The generator must run from the live source directory or the job fingerprint snapshot.');
    end
    snapshot_manifest_path = fullfile(snapshot_root, 'snapshot_manifest.json');
    if ~isfile(snapshot_manifest_path)
        error('T08:SnapshotManifestMissing', 'The source snapshot manifest is missing.');
    end
    try
        snapshot_manifest = jsondecode(fileread(snapshot_manifest_path));
    catch
        error('T08:SnapshotManifestInvalid', 'The source snapshot manifest is not valid JSON.');
    end
    if ~isstruct(snapshot_manifest) || ...
            ~isfield(snapshot_manifest, 'generation_fingerprint') || ...
            ~(ischar(snapshot_manifest.generation_fingerprint) || ...
              (isstring(snapshot_manifest.generation_fingerprint) && ...
               isscalar(snapshot_manifest.generation_fingerprint))) || ...
            ~strcmpi(char(snapshot_manifest.generation_fingerprint), config_hash)
        error('T08:SnapshotManifestMismatch', ...
            'The snapshot manifest fingerprint does not match job.config_hash.');
    end
    repo_root = project_root;
end
if ~isfield(job, 'formal_test') || ~isequal(job.formal_test, false)
    error('T08:FormalTestForbidden', 'The development generator rejects formal-test jobs.');
end
if ~any(strcmp(job.split, {'train','validation','qa_only'}))
    error('T08:ForbiddenSplit', 'Only train, validation, and qa_only development groups are allowed.');
end
if job.n_users ~= 4 || job.n_tx ~= 8 || job.n_rx_per_user ~= 1 || ...
        ~strcmp(job.topology, 'downlink_mu_miso_bs8_ue1')
    error('T08:TopologyMismatch', 'The job must use one BS with 8 Tx elements and 4 single-Rx users.');
end
if ~isfield(job, 'bs_position_m')
    error('T08:MissingBsPosition', 'The group job must specify its scenario-specific BS position.');
end
bs_position = double(job.bs_position_m(:));
if numel(bs_position) ~= 3 || any(~isfinite(bs_position))
    error('T08:InvalidBsPosition', 'The scenario-specific BS position must be a finite XYZ vector.');
end
if ~isfield(job, 'use_absolute_delays') || ~isequal(job.use_absolute_delays, true)
    error('T08:RelativeDelaysForbidden', 'Absolute propagation delays must be enabled.');
end
if strcmp(job.split, 'qa_only')
    if ~startsWith(job.scenario, '3GPP_38.901_UMi_') || ...
            ~strcmp(job.permitted_use, 'physical_geometry_time_array_file_QA_only')
        error('T08:QaOnlyScopeViolation', 'qa_only is reserved for the restricted UMi physical/file QA set.');
    end
    required_forbidden = {'model_prediction_error','selection','early_stopping', ...
        'normalization','fitting','training'};
    if ~all(ismember(required_forbidden, normalize_text_list(job.forbidden_use)))
        error('T08:QaOnlyUseLeakage', 'UMi qa_only dropped a frozen model-use exclusion.');
    end
    policy = job.qa_policy;
    if ~isfield(policy, 'physical_geometry_time_array_file_QA') || ...
            ~isequal(policy.physical_geometry_time_array_file_QA, true)
        error('T08:QaOnlyPolicyMissing', 'UMi qa_only must allow only physical/geometry/time/array/file QA.');
    end
    for field = {'model_prediction_error','model_selection','early_stopping', ...
            'normalization_fit','fitting','training'}
        if ~isfield(policy, field{1}) || ~isequal(policy.(field{1}), false)
            error('T08:QaOnlyPolicyLeakage', 'UMi qa_only policy must explicitly prohibit %s.', field{1});
        end
    end
end

output_path = canonical_path(job.output_path);
expected_path = canonical_path(fullfile(repo_root, 'revision', 'data', ...
    'development', job.split, [job.environment_id, '.mat']));
revision_data = canonical_path(fullfile(repo_root, 'revision', 'data'));
if ~strcmpi(output_path, expected_path) || ~startsWith(lower(output_path), [lower(revision_data), filesep])
    error('T08:OutputPathEscape', 'The MAT output must be the expected file below revision/data/development.');
end
if isfile(output_path)
    error('T08:RefuseOverwrite', 'The per-group output already exists; refusing to overwrite it.');
end

if ~isfield(job, 'quadriga_source') || ~isfolder(job.quadriga_source)
    error('T08:MissingQuaDRiGa', 'The pinned local QuaDRiGa source directory is unavailable.');
end
addpath(genpath(job.quadriga_source), '-begin');
simpar = qd_simulation_parameters;
simpar.center_frequency = double(job.fc_hz);
simpar.sample_density = double(job.sample_density);
simpar.use_absolute_delays = true;
simpar.show_progress_bars = false;

% BS array: eight identical omni elements on centered local-y half-wavelength spacing.
lambda = 299792458 / double(job.fc_hz);
tx_array = qd_arrayant('omni');
tx_array.copy_element(1, 2:8);
tx_array.element_position = [zeros(1,8); ((0:7)-3.5) * lambda/2; zeros(1,8)];
if size(tx_array.element_position,2) ~= 8 || ...
        max(abs(diff(tx_array.element_position(2,:)) - lambda/2)) > 1e-12 || ...
        abs(mean(tx_array.element_position(2,:))) > 1e-12
    error('T08:TxArrayLayout', 'The BS array is not a centered local-y half-wavelength ULA.');
end

time_s = double(job.time_s(:).');
n_time = numel(time_s);
if n_time < 2 || abs(time_s(1)) > 1e-12 || ...
        any(abs(diff(time_s) - double(job.dt_s)) > 1e-12)
    error('T08:TimeGrid', 'The requested target time grid is not the frozen uniform grid.');
end
if numel(job.trajectories) ~= 4
    error('T08:TrajectoryCount', 'Exactly four synchronized user trajectories are required.');
end

tracks = qd_track;
geometry_diagnostics = cell(1,4);
target_snapshot_count_by_user = zeros(1,4);
dense_snapshot_count_by_user = zeros(1,4);
max_source_gap_m_by_user = zeros(1,4);
max_dense_gap_m_by_user = zeros(1,4);
original_knot_position_error_m_by_user = zeros(1,4);
plan_distance_roundoff_m_by_user = zeros(1,4);
plan_distance_roundoff_bound_m_by_user = zeros(1,4);
movement_profile_roundoff_bound_m_by_user = zeros(1,4);
movement_profile_endpoint_clamp_bound_m_by_user = zeros(1,4);
movement_profile_endpoint_clamp_applied_by_user = false(1,4);
movement_profile_endpoint_delta_m_by_user = zeros(1,4);
source_chord_distance_m_by_user = zeros(1,4);
analytic_arc_distance_m_by_user = zeros(1,4);
analytic_arc_minus_chord_m_by_user = zeros(1,4);
target_time_position_error_m_by_user = zeros(1,4);
target_time_distance_error_m_by_user = zeros(1,4);
for u = 1:4
    trajectory = job.trajectories(u);
    pos = double(trajectory.position_m);
    vel = double(trajectory.velocity_mps);
    acc = double(trajectory.acceleration_mps2);
    orient_ypr = double(trajectory.orientation_rad);
    distance = double(trajectory.distance_m(:).');
    analytic_distance = double(trajectory.analytic_distance_m(:).');
    if ~isequal(size(pos), [n_time,3]) || ~isequal(size(vel), [n_time,3]) || ...
            ~isequal(size(acc), [n_time,3]) || ~isequal(size(orient_ypr), [n_time,3]) || ...
            numel(distance) ~= n_time || numel(analytic_distance) ~= n_time || ...
            any(diff(distance) <= 0) || any(diff(analytic_distance) <= 0)
        error('T08:TrajectoryShape', 'Each user must provide synchronized finite kinematics and distance.');
    end
    if any(~isfinite(pos(:))) || any(~isfinite(vel(:))) || any(~isfinite(acc(:))) || ...
            any(~isfinite(orient_ypr(:))) || any(~isfinite(distance)) || ...
            any(~isfinite(analytic_distance))
        error('T08:TrajectoryFinite', 'Trajectory input contains non-finite values.');
    end

    track = qd_track;
    track.name = char(trajectory.user_id);
    track.initial_position = pos(1,:).';
    track.positions = (pos - pos(1,:)).';
    % qd_track.orientation is [roll,pitch,yaw], while the frozen MAT interface is [yaw,pitch,roll].
    track.orientation = orient_ypr(:,[3,2,1]).';
    track.scenario = {char(job.scenario_segments(1).scenario)};

    % Keep every planned target-time knot and only subdivide source edges that exceed
    % the spatial sampling limit. Distances returned here are measured from that track.
    [track, target_distance_m, geometry_diagnostics{u}] = ...
        densify_qd_track_preserving_knots( ...
            track, simpar.samples_per_meter, distance);
    diagnostic = geometry_diagnostics{u};
    target_snapshot_count_by_user(u) = diagnostic.target_snapshot_count;
    dense_snapshot_count_by_user(u) = diagnostic.dense_snapshot_count;
    max_source_gap_m_by_user(u) = diagnostic.max_source_gap_m;
    max_dense_gap_m_by_user(u) = diagnostic.max_dense_gap_m;
    original_knot_position_error_m_by_user(u) = ...
        diagnostic.max_original_target_position_error_m;
    plan_distance_roundoff_m_by_user(u) = diagnostic.max_plan_distance_roundoff_m;
    plan_distance_roundoff_bound_m_by_user(u) = diagnostic.distance_roundoff_bound_m;
    movement_profile_roundoff_bound_m_by_user(u) = ...
        diagnostic.movement_profile_roundoff_bound_m;
    movement_profile_endpoint_clamp_bound_m_by_user(u) = ...
        diagnostic.movement_profile_endpoint_clamp_bound_m;
    source_chord_distance_m_by_user(u) = target_distance_m(end);
    analytic_arc_distance_m_by_user(u) = analytic_distance(end);
    analytic_arc_minus_chord_m_by_user(u) = ...
        analytic_distance(end) - target_distance_m(end);

    % Segment changes belong to the continuous track; do not splice independent CSI records.
    for k = 2:numel(job.scenario_segments)
        switch_time = double(job.scenario_segments(k).start_time_s);
        if switch_time <= 0 || switch_time >= time_s(end)
            error('T08:SegmentTime', 'Scenario transition must lie inside the trajectory time interval.');
        end
        [~, sample_idx] = min(abs(time_s - switch_time));
        switch_time_tolerance = 64 * eps(max(1,abs(switch_time)));
        if abs(time_s(sample_idx) - switch_time) > switch_time_tolerance
            error('T08:SegmentNotOnTargetKnot', ...
                'Scenario transition must start on an original target-time knot.');
        end
        dense_sample_idx = diagnostic.target_knot_index(sample_idx);
        transition_position = track.initial_position + track.positions(:,dense_sample_idx);
        track.add_segment(transition_position, char(job.scenario_segments(k).scenario));
        if ~any(track.segment_index == dense_sample_idx)
            error('T08:SegmentIndexMapping', ...
                'Scenario transition did not attach to its preserved target-time knot.');
        end
    end
    if u == 1
        tracks = track;
    else
        tracks(1,u) = track;
    end
end

layout = qd_layout(simpar);
layout.no_tx = 1;
layout.tx_position = bs_position;
layout.tx_array = tx_array;
layout.no_rx = 4;
layout.rx_array = qd_arrayant('omni');
layout.rx_track = tracks;
layout.set_pairing('all');
layout.update_rate = []; % Avoid the local get_channels(update_rate) variable-speed branch.
rng(double(job.seed), 'twister');
[channels, ~] = layout.get_channels(1, double(job.segment_overlap), char(job.interpolation_method));
if numel(channels) ~= 4
    error('T08:LinkCount', 'Expected exactly one BS-to-user channel per user.');
end

% QuaDRiGa time interpolation turns one dense spatial channel into the frozen target grid.
carriers = double(job.subcarrier_indices(:).') / double(job.n_subcarriers);
bandwidth_hz = double(job.n_subcarriers) * double(job.subcarrier_spacing_hz);
H = complex(zeros(n_time, 4, 8, double(job.n_subcarriers), 'single'));
position_m = zeros(n_time,4,3,'double');
velocity_mps = zeros(n_time,4,3,'double');
acceleration_mps2 = zeros(n_time,4,3,'double');
orientation_rad = zeros(n_time,4,3,'double');
path_power_cell = cell(1,4);
path_delay_cell = cell(1,4);
path_count = zeros(1,4);

for u = 1:4
    trajectory = job.trajectories(u);
    diagnostic = geometry_diagnostics{u};
    input_distance = double(diagnostic.target_knot_distance_m(:).');
    [dense_track_length_m, ~] = layout.rx_track(u).get_length();
    endpoint_delta_m = input_distance(end) - double(dense_track_length_m);
    movement_profile_endpoint_delta_m_by_user(u) = endpoint_delta_m;
    endpoint_bound_m = diagnostic.movement_profile_endpoint_clamp_bound_m;
    if abs(endpoint_delta_m) > endpoint_bound_m
        error('T08:MovementProfileEndpoint', ...
            'Movement-profile endpoint differs from dense track by more than roundoff.');
    elseif endpoint_delta_m ~= 0
        input_distance(end) = double(dense_track_length_m);
        movement_profile_endpoint_clamp_applied_by_user(u) = true;
    end
    movement_profile = [time_s; input_distance];
    [dist, interpolated_track] = layout.rx_track(u).interpolate( ...
        'time', double(job.dt_s), movement_profile, char(job.interpolation_method), false);
    dist = double(dist(:).');
    geometry_tolerance_m = max(1e-9, 4 * diagnostic.movement_profile_roundoff_bound_m);
    if numel(dist) ~= n_time || max(abs(dist - input_distance)) > geometry_tolerance_m
        error('T08:TimeDistanceAlignment', 'QuaDRiGa time sampling did not reproduce the frozen time-distance grid.');
    end
    interpolated_position = interpolated_track.initial_position + interpolated_track.positions;
    expected_position = double(trajectory.position_m).';
    if ~isequal(size(interpolated_position), size(expected_position)) || ...
            max(vecnorm(interpolated_position - expected_position, 2, 1)) > geometry_tolerance_m
        error('T08:PositionAlignment', 'Interpolated track positions diverge from the planned geometry.');
    end
    target_time_position_error_m_by_user(u) = ...
        max(vecnorm(interpolated_position - expected_position, 2, 1));
    target_time_distance_error_m_by_user(u) = max(abs(dist - input_distance));
    geometry_diagnostics{u}.target_knot_distance_m = input_distance;
    geometry_diagnostics{u}.max_target_time_position_error_m = ...
        target_time_position_error_m_by_user(u);
    geometry_diagnostics{u}.max_target_time_distance_error_m = ...
        target_time_distance_error_m_by_user(u);
    geometry_diagnostics{u}.movement_profile_endpoint_delta_m = endpoint_delta_m;
    geometry_diagnostics{u}.movement_profile_endpoint_clamp_applied = ...
        movement_profile_endpoint_clamp_applied_by_user(u);
    geometry_diagnostics{u}.scenario_segment_index = ...
        double(layout.rx_track(u).segment_index(:).');

    % Match links by their physical first position; never infer user order from an axis transpose.
    first_position = double(trajectory.position_m(1,:)).';
    candidate_links = [];
    for c = 1:numel(channels)
        rx_pos = channels(c).rx_position;
        if ~isempty(rx_pos) && norm(double(rx_pos(:,1)) - first_position) < 1e-5
            candidate_links(end+1) = c; %#ok<AGROW>
        end
    end
    if numel(candidate_links) ~= 1
        error('T08:LinkIdentity', 'Could not uniquely map the generated channel to its user track.');
    end
    spatial_channel = channels(candidate_links(1));
    sampled_channel = spatial_channel.interpolate(dist, char(job.interpolation_method));
    frequency_response = sampled_channel.fr(bandwidth_hz, carriers, [], false);
    if ~isequal(size(frequency_response), [1,8,double(job.n_subcarriers),n_time])
        error('T08:FrequencyResponseShape', 'QuaDRiGa frequency response axes do not match [Rx,Tx,F,T].');
    end
    H(:,u,:,:) = single(permute(frequency_response, [4,1,2,3]));

    position_m(:,u,:) = reshape(double(trajectory.position_m), [n_time,1,3]);
    velocity_mps(:,u,:) = reshape(double(trajectory.velocity_mps), [n_time,1,3]);
    acceleration_mps2(:,u,:) = reshape(double(trajectory.acceleration_mps2), [n_time,1,3]);
    orientation_rad(:,u,:) = reshape(double(trajectory.orientation_rad), [n_time,1,3]);

    coefficients = sampled_channel.coeff; % [Rx,Tx,path,time]
    delays = sampled_channel.delay;
    n_path = size(coefficients,3);
    path_count(u) = n_path;
    power = zeros(n_time,n_path,'double');
    delay = zeros(n_time,n_path,'double');
    for t_idx = 1:n_time
        for path_idx = 1:n_path
            power(t_idx,path_idx) = sum(abs(coefficients(:,:,path_idx,t_idx)).^2, 'all');
        end
    end
    if ismatrix(delays) && size(delays,1) == n_path && size(delays,2) == n_time
        delay = double(delays.');
    elseif ndims(delays) == 4 && size(delays,3) == n_path && size(delays,4) == n_time
        delay_link = reshape(double(delays), [], n_path, n_time);
        for t_idx = 1:n_time
            delay(t_idx,:) = mean(delay_link(:,:,t_idx), 1);
        end
    else
        error('T08:DelayShape', 'QuaDRiGa delay tensor has an unsupported topology.');
    end
    path_power_cell{u} = power;
    path_delay_cell{u} = delay;
end

max_paths = max(path_count);
path_power = zeros(n_time,4,max_paths,'double');
path_delay_s = nan(n_time,4,max_paths,'double');
path_valid = false(n_time,4,max_paths);
for u = 1:4
    count = path_count(u);
    path_power(:,u,1:count) = path_power_cell{u};
    path_delay_s(:,u,1:count) = path_delay_cell{u};
    path_valid(:,u,1:count) = true;
end

metadata = struct();
metadata.schema_version = '1.0';
metadata.dataset = 'development';
metadata.environment_id = char(job.environment_id);
metadata.realization_id = char(job.realization_id);
metadata.user_group_id = char(job.user_group_id);
metadata.split = char(job.split);
metadata.topology = char(job.topology);
metadata.scenario = char(job.scenario);
metadata.bs_position_m = reshape(bs_position, 1, 3);
metadata.speed_kmh = double(job.speed_kmh);
metadata.trajectory = char(job.trajectory);
metadata.seed = double(job.seed);
metadata.rng = struct('generator','MATLAB twister','seed',double(job.seed));
metadata.generator_version = 't08-dev-1';
metadata.config_hash = char(job.config_hash);
metadata.array = struct('n_tx',8,'n_rx_per_user',1,'n_users',4, ...
    'element_pattern','qd_arrayant omni; identical vertically polarized elements', ...
    'axis','local_y','spacing','half_wavelength','centered',true, ...
    'bs_yaw_pitch_roll_rad',[0,0,0]);
metadata.delay_reference = struct('use_absolute_delays',true,'unit','s', ...
    'reference','absolute propagation delay');
scenario_segment_index_by_user = cell(1,4);
for u = 1:4
    scenario_segment_index_by_user{u} = ...
        double(layout.rx_track(u).segment_index(:).');
end
metadata.source_grid = struct('sample_density',double(job.sample_density), ...
    'samples_per_meter',double(simpar.samples_per_meter), ...
    'maximum_allowed_spatial_gap_m',1/double(simpar.samples_per_meter), ...
    'original_target_time_knots_preserved',true, ...
    'source_edges_only_subdivided',true, ...
    'target_snapshot_count_by_user',target_snapshot_count_by_user, ...
    'dense_snapshot_count_by_user',dense_snapshot_count_by_user, ...
    'max_source_gap_m_by_user',max_source_gap_m_by_user, ...
    'max_dense_gap_m_by_user',max_dense_gap_m_by_user, ...
    'original_target_knot_position_error_max_m_by_user',original_knot_position_error_m_by_user, ...
    'target_time_position_error_max_m_by_user',target_time_position_error_m_by_user, ...
    'target_time_distance_error_max_m_by_user',target_time_distance_error_m_by_user, ...
    'plan_distance_roundoff_max_m_by_user',plan_distance_roundoff_m_by_user, ...
    'plan_distance_roundoff_bound_m_by_user',plan_distance_roundoff_bound_m_by_user, ...
    'movement_profile_roundoff_bound_m_by_user',movement_profile_roundoff_bound_m_by_user, ...
    'movement_profile_endpoint_delta_m_by_user',movement_profile_endpoint_delta_m_by_user, ...
    'movement_profile_endpoint_clamp_bound_m_by_user',movement_profile_endpoint_clamp_bound_m_by_user, ...
    'movement_profile_endpoint_clamp_applied_by_user',movement_profile_endpoint_clamp_applied_by_user, ...
    'source_chord_distance_m_by_user',source_chord_distance_m_by_user, ...
    'analytic_arc_distance_m_by_user',analytic_arc_distance_m_by_user, ...
    'analytic_arc_minus_source_chord_m_by_user',analytic_arc_minus_chord_m_by_user, ...
    'scenario_segment_start_index_by_user',{scenario_segment_index_by_user}, ...
    'quadri_ga_version',qd_simulation_parameters.version);
metadata.interpolation = struct('route','dense_spatial_channels_then_qd_track_time_and_qd_channel_interpolation', ...
    'track_geometry','all target-time position and orientation knots are retained; source edges are linearly subdivided only when longer than the maximum spatial gap', ...
    'track_method','time','channel_method',char(job.interpolation_method), ...
    'target_dt_s',double(job.dt_s),'segment_overlap',double(job.segment_overlap), ...
    'path_count_by_user',path_count,'path_index_caveat', ...
    'path IDs are not stable across scenario segments; no independent CSI splicing');
metadata.synthetic_fixture = false;
metadata.permitted_use = char(job.permitted_use);
metadata.forbidden_use = normalize_text_list(job.forbidden_use);
if strcmp(job.split, 'qa_only')
    metadata.qa_policy = job.qa_policy;
else
    metadata.qa_policy = struct();
end
metadata.family = char(job.family);
metadata.condition = char(job.condition);
metadata.formal_test = false;
meta_json = jsonencode(metadata);

output_parent = fileparts(output_path);
if ~isfolder(output_parent)
    mkdir(output_parent);
end
temp_path = [output_path, '.tmp.mat'];
if isfile(temp_path)
    error('T08:TemporaryOutputExists', 'A previous temporary output exists; refusing to overwrite it.');
end
save(temp_path, 'H', 'time_s', 'position_m', 'velocity_mps', ...
    'acceleration_mps2', 'orientation_rad', 'path_power', 'path_delay_s', ...
    'path_valid', 'meta_json', '-v7');
[ok, message] = movefile(temp_path, output_path);
if ~ok
    error('T08:AtomicMoveFailed', 'Could not commit MAT output: %s', message);
end
fprintf('GROUP_GENERATED=%s\n', job.environment_id);
fprintf('OUTPUT=%s\n', output_path);
fprintf('H_SHAPE=%s\n', mat2str(size(H)));
fprintf('PATH_COUNTS=%s\n', mat2str(path_count));
fprintf('REAL_SCIENTIFIC_CHANNEL=true\n');
end

function out = canonical_path(path)
out = char(java.io.File(char(path)).getCanonicalPath());
end

function values = normalize_text_list(value)
if isempty(value)
    values = {};
elseif iscell(value)
    values = cellfun(@char, value, 'UniformOutput', false);
elseif isstring(value)
    values = cellstr(value(:));
elseif ischar(value)
    values = cellstr(value);
else
    error('T08:InvalidTextList', 'Expected a JSON text array.');
end
end
