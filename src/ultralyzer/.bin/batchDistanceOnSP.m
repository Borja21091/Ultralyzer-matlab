function distances = batchDistanceOnSP(xs, ys, xf, yf)
% batchDistanceOnSP Compute many distanceOnSP calls in one MATLAB round-trip.
%
% This wrapper accepts same-shaped numeric arrays, converts them to scalar
% calls into distanceOnSP, and reshapes the distance output back to the
% original input shape. It is used from Python to avoid one engine call per
% segment.

validateattributes(xs, {'numeric'}, {'real'}, mfilename, 'xs', 1);
validateattributes(ys, {'numeric'}, {'real'}, mfilename, 'ys', 2);
validateattributes(xf, {'numeric'}, {'real'}, mfilename, 'xf', 3);
validateattributes(yf, {'numeric'}, {'real'}, mfilename, 'yf', 4);

if ~isequal(size(xs), size(ys)) || ~isequal(size(xs), size(xf)) || ~isequal(size(xs), size(yf))
    error('batchDistanceOnSP:sizeMismatch', ...
        'Inputs xs, ys, xf, and yf must have identical sizes.');
end

original_size = size(xs);
xs = double(xs(:).');
ys = double(ys(:).');
xf = double(xf(:).');
yf = double(yf(:).');

distances = zeros(1, numel(xs));
for idx = 1:numel(xs)
    distances(idx) = distanceOnSP(xs(idx), ys(idx), xf(idx), yf(idx));
end

distances = reshape(distances, original_size);
end