import numpy as np
from scipy.optimize import linear_sum_assignment
from core.utils.kalmanfilter import KalmanFilter


class Track():
    def __init__(self, id):
        self.id = id
        self.filter = KalmanFilter(0.1, 1, 1, 1, 0.1, 0.1)

    def update(self, point):
        return self.filter.update([[point[0]], [point[1]]])

    def predict(self):
        x, y = self.filter.predict()
        return x, y


class Tracker():
    def __init__(self, dist_thresh):
        self.tracks = []
        self.dist_thresh = dist_thresh
        self.counter = 0

    def update(self, points):
        # Match points and tracks
        matrix = []
        for i, _ in enumerate(self.tracks):
            for j, _ in enumerate(points):
                if len(matrix) < i+1:
                    matrix.append([])
                if len(matrix[i]) < j+1:
                    matrix[i].append([])
                (x, y) = self.tracks[i].predict()
                matrix[i][j] = np.linalg.norm([x - points[j][0], y - points[j][1]])

        points_id_matched = []
        if matrix:
            row_ind, col_ind = linear_sum_assignment(np.array(matrix))
            for track_id, point_id in zip(row_ind, col_ind):
                dist = matrix[track_id][point_id]
                if dist <= self.dist_thresh:
                    self.tracks[track_id].update(points[point_id])
                    points_id_matched.append(point_id)

        # Create new tracks
        for id, _ in enumerate(points):
            if id not in points_id_matched:
                new_track = Track(self.counter)
                new_track.update(points[id])
                self.tracks.append(new_track)
                self.counter += 1







