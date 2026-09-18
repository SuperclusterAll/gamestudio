extends Node2D

# ── Constants ──────────────────────────────────────────────────────────────────
const TILE_SIZE : int = 20
const COLS : int = 28
const ROWS : int = 29
const MAZE_OFFSET_X : int = 0
const MAZE_OFFSET_Y : int = 40

const PLAYER_SPEED : float = 120.0
const GHOST_SPEED : float = 80.0
const POWER_DURATION : float = 8.0
const INVINCIBLE_DURATION : float = 2.0
const PATROL_DURATION : float = 15.0
const FLASH_START : float = 2.0

const DOT_SCORE : int = 10
const GHOST_KILL_SCORE : int = 50

# 1=wall, 0=dot, 2=power pellet, 3=empty(no dot), 4=ghost house
const MAZE_DATA : Array = [
	[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
	[1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1],
	[1,0,1,1,1,1,0,1,1,1,1,1,0,1,1,0,1,1,1,1,1,0,1,1,1,1,0,1],
	[1,2,1,1,1,1,0,1,1,1,1,1,0,1,1,0,1,1,1,1,1,0,1,1,1,1,2,1],
	[1,0,1,1,1,1,0,1,1,1,1,1,0,1,1,0,1,1,1,1,1,0,1,1,1,1,0,1],
	[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
	[1,0,1,1,1,1,0,1,1,0,1,1,1,1,1,1,1,1,0,1,1,0,1,1,1,1,0,1],
	[1,0,1,1,1,1,0,1,1,0,1,1,1,1,1,1,1,1,0,1,1,0,1,1,1,1,0,1],
	[1,0,0,0,0,0,0,1,1,0,0,0,0,1,1,0,0,0,0,1,1,0,0,0,0,0,0,1],
	[1,1,1,1,1,1,0,1,1,1,1,1,3,1,1,3,1,1,1,1,1,0,1,1,1,1,1,1],
	[1,1,1,1,1,1,0,1,1,3,3,3,3,3,3,3,3,3,3,1,1,0,1,1,1,1,1,1],
	[1,1,1,1,1,1,0,1,1,3,1,1,1,4,4,1,1,1,3,1,1,0,1,1,1,1,1,1],
	[1,1,1,1,1,1,0,1,1,3,1,4,4,4,4,4,4,1,3,1,1,0,1,1,1,1,1,1],
	[3,3,3,3,3,3,0,3,3,3,1,4,4,4,4,4,4,1,3,3,3,0,3,3,3,3,3,3],
	[1,1,1,1,1,1,0,1,1,3,1,4,4,4,4,4,4,1,3,1,1,0,1,1,1,1,1,1],
	[1,1,1,1,1,1,0,1,1,3,1,1,1,1,1,1,1,1,3,1,1,0,1,1,1,1,1,1],
	[1,1,1,1,1,1,0,1,1,3,3,3,3,3,3,3,3,3,3,1,1,0,1,1,1,1,1,1],
	[1,1,1,1,1,1,0,1,1,3,1,1,1,1,1,1,1,1,3,1,1,0,1,1,1,1,1,1],
	[1,0,0,0,0,0,0,0,0,0,0,0,0,1,1,0,0,0,0,0,0,0,0,0,0,0,0,1],
	[1,0,1,1,1,1,0,1,1,1,1,1,0,1,1,0,1,1,1,1,1,0,1,1,1,1,0,1],
	[1,0,1,1,1,1,0,1,1,1,1,1,0,1,1,0,1,1,1,1,1,0,1,1,1,1,0,1],
	[1,2,0,0,1,1,0,0,0,0,0,0,0,3,3,0,0,0,0,0,0,0,1,1,0,0,2,1],
	[1,1,1,0,1,1,0,1,1,0,1,1,1,1,1,1,1,1,0,1,1,0,1,1,0,1,1,1],
	[1,1,1,0,1,1,0,1,1,0,1,1,1,1,1,1,1,1,0,1,1,0,1,1,0,1,1,1],
	[1,0,0,0,0,0,0,1,1,0,0,0,0,1,1,0,0,0,0,1,1,0,0,0,0,0,0,1],
	[1,0,1,1,1,1,1,1,1,1,1,1,0,1,1,0,1,1,1,1,1,1,1,1,1,1,0,1],
	[1,0,1,1,1,1,1,1,1,1,1,1,0,1,1,0,1,1,1,1,1,1,1,1,1,1,0,1],
	[1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1],
	[1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1],
]

enum GameState { PLAYING, GHOST_HUNT, HIT, GAME_OVER, CLEAR }
enum GhostMode { PATROL, CHASE, SCARED, RETURNING }

@onready var score_label: Label = $UI/TopBar/ScoreLabel
@onready var lives_label: Label = $UI/TopBar/LivesLabel
@onready var timer_label: Label = $UI/TopBar/TimerLabel
@onready var overlay_label: Label = $UI/OverlayLabel

var tex_player: Texture2D = null
var tex_ghosts: Array = []
var tex_ghost_scared: Texture2D = null
var tex_ghost_flash: Texture2D = null

var game_state: int = GameState.PLAYING
var score: int = 0
var lives: int = 3
var elapsed_time: float = 0.0
var power_timer: float = 0.0
var invincible_timer: float = 0.0
var patrol_timer: float = 0.0
var ghosts_killed: int = 0
var waiting_for_key: bool = false

var dots: Array = []
var total_dots: int = 0
var eaten_dots: int = 0

const PLAYER_START_COL : int = 13
const PLAYER_START_ROW : int = 21

var player_col: int = PLAYER_START_COL
var player_row: int = PLAYER_START_ROW
var player_pos: Vector2 = Vector2.ZERO
var player_dir: Vector2 = Vector2.ZERO
var player_next_dir: Vector2 = Vector2.ZERO
var mouth_open: bool = true
var mouth_timer: float = 0.0

var ghosts: Array = []
const GHOST_COLORS : Array = [Color(1,0,0), Color(1,0.72,1), Color(0,1,1), Color(1,0.72,0.32)]

func get_tile(col: int, row: int) -> int:
	if col < 0 or col >= COLS or row < 0 or row >= ROWS:
		return 1
	var row_data: Array = MAZE_DATA[row]
	return int(row_data[col])

func is_walkable(col: int, row: int) -> bool:
	var t: int = get_tile(col, row)
	return t == 0 or t == 2 or t == 3 or t == 4

func tile_to_pixel(col: int, row: int) -> Vector2:
	return Vector2(
		float(MAZE_OFFSET_X + col * TILE_SIZE + TILE_SIZE / 2),
		float(MAZE_OFFSET_Y + row * TILE_SIZE + TILE_SIZE / 2)
	)

func pixel_to_tile(pos: Vector2) -> Vector2i:
	return Vector2i(
		int((pos.x - float(MAZE_OFFSET_X)) / float(TILE_SIZE)),
		int((pos.y - float(MAZE_OFFSET_Y)) / float(TILE_SIZE))
	)

func bfs_next_dir(from_col: int, from_row: int, to_col: int, to_row: int) -> Vector2:
	if from_col == to_col and from_row == to_row:
		return Vector2.ZERO
	var visited: Dictionary = {}
	var queue: Array = []
	var dirs: Array = [Vector2i(1,0), Vector2i(-1,0), Vector2i(0,1), Vector2i(0,-1)]
	queue.append([from_col, from_row, Vector2.ZERO])
	visited[Vector2i(from_col, from_row)] = true
	while queue.size() > 0:
		var cur: Array = queue.pop_front()
		var cc: int = int(cur[0])
		var cr: int = int(cur[1])
		var first_dir: Vector2 = cur[2]
		for dv in dirs:
			var d: Vector2i = dv
			var nc: int = cc + d.x
			var nr: int = cr + d.y
			if nc < 0:
				nc = COLS - 1
			elif nc >= COLS:
				nc = 0
			var key: Vector2i = Vector2i(nc, nr)
			if not visited.has(key) and is_walkable(nc, nr):
				var fd: Vector2 = first_dir if first_dir != Vector2.ZERO else Vector2(float(d.x), float(d.y))
				if nc == to_col and nr == to_row:
					return fd
				visited[key] = true
				queue.append([nc, nr, fd])
	return Vector2.ZERO

func bfs_next_dir_away(from_col: int, from_row: int, target_col: int, target_row: int) -> Vector2:
	var dirs: Array = [Vector2i(1,0), Vector2i(-1,0), Vector2i(0,1), Vector2i(0,-1)]
	var best_dir: Vector2 = Vector2.ZERO
	var best_dist: float = -1.0
	for dv in dirs:
		var d: Vector2i = dv
		var nc: int = from_col + d.x
		var nr: int = from_row + d.y
		if nc < 0: nc = COLS - 1
		elif nc >= COLS: nc = 0
		if is_walkable(nc, nr):
			var dist: float = Vector2(float(nc - target_col), float(nr - target_row)).length()
			if dist > best_dist:
				best_dist = dist
				best_dir = Vector2(float(d.x), float(d.y))
	return best_dir

func _ready() -> void:
	_load_textures()
	_init_game()

func _load_textures() -> void:
	if ResourceLoader.exists("res://assets/player-normal.png"):
		tex_player = load("res://assets/player-normal.png")
	tex_ghosts.resize(4)
	var paths: Array = ["res://assets/ghost-red.png","res://assets/ghost-pink.png","res://assets/ghost-cyan.png","res://assets/ghost-orange.png"]
	for i in range(4):
		if ResourceLoader.exists(paths[i]):
			tex_ghosts[i] = load(paths[i])
		else:
			tex_ghosts[i] = null
	if ResourceLoader.exists("res://assets/ghost-scared.png"):
		tex_ghost_scared = load("res://assets/ghost-scared.png")
	if ResourceLoader.exists("res://assets/ghost-scared-flash.png"):
		tex_ghost_flash = load("res://assets/ghost-scared-flash.png")

func _init_game() -> void:
	score = 0
	lives = 3
	elapsed_time = 0.0
	power_timer = 0.0
	invincible_timer = 0.0
	patrol_timer = 0.0
	ghosts_killed = 0
	eaten_dots = 0
	game_state = GameState.PLAYING
	waiting_for_key = false
	overlay_label.text = ""

	_build_dots()
	_setup_player()
	_setup_ghosts()
	_update_ui()
	queue_redraw()

func _build_dots() -> void:
	dots.clear()
	total_dots = 0
	for row in range(ROWS):
		for col in range(COLS):
			var row_data: Array = MAZE_DATA[row]
			var t: int = int(row_data[col])
			if t == 0 or t == 2:
				var is_power: bool = (t == 2)
				dots.append([col, row, is_power, false])
				if not is_power:
					total_dots += 1

func _setup_player() -> void:
	player_col = PLAYER_START_COL
	player_row = PLAYER_START_ROW
	player_pos = tile_to_pixel(player_col, player_row)
	player_dir = Vector2.ZERO
	player_next_dir = Vector2.ZERO
	mouth_open = true
	mouth_timer = 0.0

func _setup_ghosts() -> void:
	ghosts.clear()
	# Ghost house center positions — all type 4 (walkable)
	var start_cols: Array = [13, 14, 12, 15]
	var start_rows: Array = [11, 11, 13, 13]
	for i in range(4):
		var sc: int = int(start_cols[i])
		var sr: int = int(start_rows[i])
		# Each ghost starts at its tile center, with no direction yet
		# We give them an initial direction so they start moving immediately
		var init_dir: Vector2 = Vector2(0.0, -1.0)  # start moving up
		ghosts.append({
			"col": sc,
			"row": sr,
			"pos": tile_to_pixel(sc, sr),
			"dir": init_dir,
			"next_col": sc,
			"next_row": sr - 1,
			"at_center": false,
			"mode": GhostMode.PATROL,
			"home_col": sc,
			"home_row": sr,
			"color_idx": i,
		})

func _process(delta: float) -> void:
	if waiting_for_key:
		if Input.is_action_just_pressed("ui_accept") or \
		   Input.is_action_just_pressed("move_up") or \
		   Input.is_action_just_pressed("move_down") or \
		   Input.is_action_just_pressed("move_left") or \
		   Input.is_action_just_pressed("move_right"):
			_init_game()
		return

	if game_state == GameState.GAME_OVER or game_state == GameState.CLEAR:
		return

	elapsed_time += delta
	patrol_timer += delta

	if game_state == GameState.GHOST_HUNT:
		power_timer -= delta
		if power_timer <= 0.0:
			power_timer = 0.0
			game_state = GameState.PLAYING
			for g in ghosts:
				var gd: Dictionary = g
				if gd.mode == GhostMode.SCARED:
					if patrol_timer >= PATROL_DURATION:
						gd.mode = GhostMode.CHASE
					else:
						gd.mode = GhostMode.PATROL

	if invincible_timer > 0.0:
		invincible_timer -= delta
		if invincible_timer < 0.0:
			invincible_timer = 0.0

	_handle_input()
	_move_player(delta)
	_check_dot_pickup()
	_move_ghosts(delta)
	_check_ghost_collision()
	_update_ui()

	mouth_timer += delta
	if mouth_timer > 0.15:
		mouth_timer = 0.0
		mouth_open = not mouth_open

	queue_redraw()

func _handle_input() -> void:
	if Input.is_action_pressed("move_up"):
		player_next_dir = Vector2(0.0, -1.0)
	elif Input.is_action_pressed("move_down"):
		player_next_dir = Vector2(0.0, 1.0)
	elif Input.is_action_pressed("move_left"):
		player_next_dir = Vector2(-1.0, 0.0)
	elif Input.is_action_pressed("move_right"):
		player_next_dir = Vector2(1.0, 0.0)

func _move_player(delta: float) -> void:
	# Try to apply buffered direction change when close to tile center
	if player_next_dir != Vector2.ZERO and player_next_dir != player_dir:
		var next_col: int = player_col + int(player_next_dir.x)
		var next_row: int = player_row + int(player_next_dir.y)
		var center: Vector2 = tile_to_pixel(player_col, player_row)
		var dist: float = (player_pos - center).length()
		if dist < float(TILE_SIZE) * 0.45 and is_walkable(next_col, next_row):
			player_dir = player_next_dir
			player_next_dir = Vector2.ZERO
			player_pos = center

	if player_dir != Vector2.ZERO:
		var target_col: int = player_col + int(player_dir.x)
		var target_row: int = player_row + int(player_dir.y)
		if target_col < 0:
			target_col = COLS - 1
		elif target_col >= COLS:
			target_col = 0

		if is_walkable(target_col, target_row):
			player_pos += player_dir * PLAYER_SPEED * delta
			var new_tile: Vector2i = pixel_to_tile(player_pos)
			if new_tile.x != player_col or new_tile.y != player_row:
				player_col = new_tile.x
				player_row = new_tile.y
				var center: Vector2 = tile_to_pixel(player_col, player_row)
				if player_dir.x != 0.0:
					player_pos.y = center.y
				else:
					player_pos.x = center.x
		else:
			var center: Vector2 = tile_to_pixel(player_col, player_row)
			player_pos = center
			player_dir = Vector2.ZERO

	# Tunnel wrap
	if player_pos.x < float(MAZE_OFFSET_X):
		player_pos.x = float(MAZE_OFFSET_X + COLS * TILE_SIZE) - float(TILE_SIZE) / 2.0
		player_col = COLS - 1
	elif player_pos.x > float(MAZE_OFFSET_X + COLS * TILE_SIZE):
		player_pos.x = float(MAZE_OFFSET_X) + float(TILE_SIZE) / 2.0
		player_col = 0

func _check_dot_pickup() -> void:
	for i in range(dots.size()):
		var d: Array = dots[i]
		if bool(d[3]):
			continue
		if int(d[0]) == player_col and int(d[1]) == player_row:
			d[3] = true
			if bool(d[2]):
				_activate_power_pellet()
			else:
				score += DOT_SCORE
				eaten_dots += 1
				if eaten_dots >= total_dots:
					_trigger_clear()

func _activate_power_pellet() -> void:
	score += DOT_SCORE
	power_timer = POWER_DURATION
	game_state = GameState.GHOST_HUNT
	for g in ghosts:
		var gd: Dictionary = g
		if gd.mode != GhostMode.RETURNING:
			gd.mode = GhostMode.SCARED

func _trigger_clear() -> void:
	game_state = GameState.CLEAR
	var final_score: int = score - int(elapsed_time * 10.0)
	if final_score < 0:
		final_score = 0
	overlay_label.text = "STAGE CLEAR!\n최종 점수: %d\n클리어 시간: %.1f초\n\n아무 키나 눌러 재시작" % [final_score, elapsed_time]
	waiting_for_key = true

# ── Ghost movement ─────────────────────────────────────────────────────────────
# Each ghost moves tile-by-tile. It always has a target tile (next_col/next_row).
# When it arrives at that tile's center, it picks the next tile and continues.
# This guarantees ghosts never get stuck between tiles.

func _move_ghosts(delta: float) -> void:
	for g in ghosts:
		var gd: Dictionary = g
		var speed: float = GHOST_SPEED
		if gd.mode == GhostMode.SCARED:
			speed = GHOST_SPEED * 0.5
		elif gd.mode == GhostMode.RETURNING:
			speed = GHOST_SPEED * 2.0

		var target_pos: Vector2 = tile_to_pixel(int(gd.next_col), int(gd.next_row))
		var to_target: Vector2 = target_pos - gd.pos
		var dist: float = to_target.length()
		var step: float = speed * delta

		if step >= dist:
			# Arrived at (or past) the target tile center — snap and pick next tile
			gd.pos = target_pos
			gd.col = gd.next_col
			gd.row = gd.next_row

			# Check if ghost returned home
			if gd.mode == GhostMode.RETURNING and int(gd.col) == int(gd.home_col) and int(gd.row) == int(gd.home_row):
				if patrol_timer >= PATROL_DURATION:
					gd.mode = GhostMode.CHASE
				else:
					gd.mode = GhostMode.PATROL

			# Pick next direction and target tile
			var new_dir: Vector2 = _choose_ghost_dir(gd)
			if new_dir == Vector2.ZERO:
				# No valid direction — try any walkable neighbour (fallback)
				new_dir = _random_walkable_dir(int(gd.col), int(gd.row), gd.dir)
			if new_dir != Vector2.ZERO:
				var nc: int = int(gd.col) + int(new_dir.x)
				var nr: int = int(gd.row) + int(new_dir.y)
				if nc < 0: nc = COLS - 1
				elif nc >= COLS: nc = 0
				gd.dir = new_dir
				gd.next_col = nc
				gd.next_row = nr
			# If still zero, ghost stays put (shouldn't happen in a well-formed maze)
		else:
			# Still travelling toward target tile center
			gd.pos = gd.pos + to_target.normalized() * step

		# Tunnel wrap
		if gd.pos.x < float(MAZE_OFFSET_X) - float(TILE_SIZE):
			gd.pos.x = float(MAZE_OFFSET_X + COLS * TILE_SIZE) - float(TILE_SIZE) / 2.0
			gd.col = COLS - 1
			gd.next_col = COLS - 1
		elif gd.pos.x > float(MAZE_OFFSET_X + COLS * TILE_SIZE):
			gd.pos.x = float(MAZE_OFFSET_X) + float(TILE_SIZE) / 2.0
			gd.col = 0
			gd.next_col = 0

func _choose_ghost_dir(gd: Dictionary) -> Vector2:
	var gc: int = int(gd.col)
	var gr: int = int(gd.row)
	if gd.mode == GhostMode.PATROL:
		return _random_walkable_dir(gc, gr, gd.dir)
	elif gd.mode == GhostMode.CHASE:
		return bfs_next_dir(gc, gr, player_col, player_row)
	elif gd.mode == GhostMode.SCARED:
		return bfs_next_dir_away(gc, gr, player_col, player_row)
	elif gd.mode == GhostMode.RETURNING:
		return bfs_next_dir(gc, gr, int(gd.home_col), int(gd.home_row))
	return Vector2.ZERO

func _random_walkable_dir(col: int, row: int, current_dir: Vector2) -> Vector2:
	var dirs: Array = [Vector2(1,0), Vector2(-1,0), Vector2(0,1), Vector2(0,-1)]
	var valid: Array = []
	var reverse: Vector2 = -current_dir
	for dv in dirs:
		var d: Vector2 = dv
		var nc: int = col + int(d.x)
		var nr: int = row + int(d.y)
		if nc < 0: nc = COLS - 1
		elif nc >= COLS: nc = 0
		if is_walkable(nc, nr) and d != reverse:
			valid.append(d)
	if valid.size() == 0:
		for dv in dirs:
			var d: Vector2 = dv
			var nc: int = col + int(d.x)
			var nr: int = row + int(d.y)
			if nc < 0: nc = COLS - 1
			elif nc >= COLS: nc = 0
			if is_walkable(nc, nr):
				valid.append(d)
	if valid.size() == 0:
		return Vector2.ZERO
	return valid[randi() % valid.size()]

func _check_ghost_collision() -> void:
	if invincible_timer > 0.0:
		return
	for g in ghosts:
		var gd: Dictionary = g
		if gd.mode == GhostMode.RETURNING:
			continue
		var dist: float = (gd.pos - player_pos).length()
		if dist < float(TILE_SIZE) * 0.8:
			if gd.mode == GhostMode.SCARED:
				gd.mode = GhostMode.RETURNING
				score += GHOST_KILL_SCORE
				ghosts_killed += 1
			else:
				_player_hit()
				return

func _player_hit() -> void:
	lives -= 1
	if lives <= 0:
		lives = 0
		game_state = GameState.GAME_OVER
		var final_score: int = score - int(elapsed_time * 10.0)
		if final_score < 0:
			final_score = 0
		overlay_label.text = "GAME OVER\n최종 점수: %d\n\n아무 키나 눌러 재시작" % [final_score]
		waiting_for_key = true
	else:
		player_col = PLAYER_START_COL
		player_row = PLAYER_START_ROW
		player_pos = tile_to_pixel(player_col, player_row)
		player_dir = Vector2.ZERO
		player_next_dir = Vector2.ZERO
		invincible_timer = INVINCIBLE_DURATION
		if game_state == GameState.GHOST_HUNT:
			power_timer = 0.0
			game_state = GameState.PLAYING
		for g in ghosts:
			var gd: Dictionary = g
			if gd.mode == GhostMode.SCARED:
				gd.mode = GhostMode.PATROL

func _update_ui() -> void:
	score_label.text = "점수: %d" % score
	var hearts: String = ""
	for i in range(lives):
		hearts += "♥"
	for i in range(3 - lives):
		hearts += "♡"
	lives_label.text = hearts
	timer_label.text = "%.1fs" % elapsed_time

func _draw() -> void:
	# Background
	draw_rect(Rect2(0.0, 0.0, 560.0, 620.0), Color(0.0, 0.0, 0.0))

	# Draw maze walls
	var wall_color: Color = Color(0.1, 0.1, 1.0)
	var wall_inner: Color = Color(0.27, 0.27, 1.0)
	for row in range(ROWS):
		for col in range(COLS):
			var row_data: Array = MAZE_DATA[row]
			var t: int = int(row_data[col])
			if t == 1:
				var rx: float = float(MAZE_OFFSET_X + col * TILE_SIZE)
				var ry: float = float(MAZE_OFFSET_Y + row * TILE_SIZE)
				draw_rect(Rect2(rx, ry, float(TILE_SIZE), float(TILE_SIZE)), wall_color)
				draw_rect(Rect2(rx + 2.0, ry + 2.0, float(TILE_SIZE) - 4.0, float(TILE_SIZE) - 4.0), wall_inner)

	# Draw dots
	for i in range(dots.size()):
		var d: Array = dots[i]
		if bool(d[3]):
			continue
		var dc: int = int(d[0])
		var dr: int = int(d[1])
		var dp: bool = bool(d[2])
		var dpos: Vector2 = tile_to_pixel(dc, dr)
		if dp:
			var pulse: float = 7.0 + 2.0 * sin(elapsed_time * 4.0)
			draw_circle(dpos, pulse, Color(1.0, 1.0, 1.0))
		else:
			draw_circle(dpos, 2.0, Color(1.0, 1.0, 1.0))

	_draw_player()
	_draw_ghosts()

func _draw_player() -> void:
	var radius: float = 9.0
	var alpha: float = 1.0
	if invincible_timer > 0.0:
		alpha = 0.5 + 0.5 * sin(elapsed_time * 20.0)
	var col: Color = Color(1.0, 0.878, 0.0, alpha)

	if tex_player != null:
		var angle: float = 0.0
		if player_dir.x > 0.0:
			angle = 0.0
		elif player_dir.x < 0.0:
			angle = PI
		elif player_dir.y > 0.0:
			angle = PI / 2.0
		elif player_dir.y < 0.0:
			angle = -PI / 2.0
		var size: Vector2 = Vector2(radius * 2.2, radius * 2.2)
		draw_set_transform(player_pos, angle)
		draw_texture_rect(tex_player, Rect2(-size / 2.0, size), false, Color(1.0, 1.0, 1.0, alpha))
		draw_set_transform(Vector2.ZERO, 0.0)
	else:
		if mouth_open:
			var mouth_angle: float = deg_to_rad(30.0)
			var base_angle: float = 0.0
			if player_dir.x < 0.0:
				base_angle = PI
			elif player_dir.y > 0.0:
				base_angle = PI / 2.0
			elif player_dir.y < 0.0:
				base_angle = -PI / 2.0
			draw_circle(player_pos, radius, col)
			var pts: PackedVector2Array = PackedVector2Array()
			pts.append(player_pos)
			var steps: int = 8
			for s in range(steps + 1):
				var a: float = base_angle - mouth_angle + (2.0 * mouth_angle * float(s) / float(steps))
				pts.append(player_pos + Vector2(cos(a), sin(a)) * (radius + 2.0))
			draw_colored_polygon(pts, Color(0.0, 0.0, 0.0))
		else:
			draw_circle(player_pos, radius, col)

func _draw_ghosts() -> void:
	var flash: bool = false
	if game_state == GameState.GHOST_HUNT and power_timer < FLASH_START:
		flash = fmod(power_timer, 0.4) < 0.2

	for g in ghosts:
		var gd: Dictionary = g
		var gpos: Vector2 = gd.pos
		var radius: float = 9.0
		var cidx: int = int(gd.color_idx)
		var gmode: int = int(gd.mode)

		if gmode == GhostMode.RETURNING:
			draw_circle(gpos + Vector2(-3.0, -2.0), 3.0, Color(1.0, 1.0, 1.0))
			draw_circle(gpos + Vector2(3.0, -2.0), 3.0, Color(1.0, 1.0, 1.0))
			draw_circle(gpos + Vector2(-2.0, -2.0), 1.5, Color(0.0, 0.0, 1.0))
			draw_circle(gpos + Vector2(4.0, -2.0), 1.5, Color(0.0, 0.0, 1.0))
			continue

		var draw_color: Color = GHOST_COLORS[cidx]
		var use_tex: Texture2D = null
		var tex_modulate: Color = Color(1.0, 1.0, 1.0)

		if gmode == GhostMode.SCARED:
			if flash:
				draw_color = Color(1.0, 1.0, 1.0)
				use_tex = tex_ghost_flash
			else:
				draw_color = Color(0.13, 0.13, 0.87)
				use_tex = tex_ghost_scared
			tex_modulate = draw_color
		else:
			use_tex = tex_ghosts[cidx]

		if use_tex != null:
			var size: Vector2 = Vector2(radius * 2.2, radius * 2.2)
			draw_texture_rect(use_tex, Rect2(gpos - size / 2.0, size), false, tex_modulate)
		else:
			draw_circle(gpos + Vector2(0.0, -2.0), radius, draw_color)
			draw_rect(Rect2(gpos.x - radius, gpos.y - 2.0, radius * 2.0, radius + 2.0), draw_color)
			draw_circle(gpos + Vector2(-3.0, -3.0), 3.5, Color(1.0, 1.0, 1.0))
			draw_circle(gpos + Vector2(3.0, -3.0), 3.5, Color(1.0, 1.0, 1.0))
			if gmode == GhostMode.SCARED:
				draw_circle(gpos + Vector2(-3.0, -3.0), 1.5, Color(0.5, 0.5, 1.0))
				draw_circle(gpos + Vector2(3.0, -3.0), 1.5, Color(0.5, 0.5, 1.0))
			else:
				var gc: Color = GHOST_COLORS[cidx]
				draw_circle(gpos + Vector2(-2.0, -3.0), 1.5, Color(gc.r * 0.5, 0.0, 0.5))
				draw_circle(gpos + Vector2(4.0, -3.0), 1.5, Color(gc.r * 0.5, 0.0, 0.5))
