use crate::numerics::FromVector2;
use crate::windows::{
    foundation::numerics::{Vector2, Vector3},
    ui::{
        composition::{CompositionBorderMode, Compositor, ContainerVisual, SpriteVisual},
        Colors,
    },
};
use rand::distributions::{Distribution, Uniform};
use std::collections::VecDeque;
use winit::event::VirtualKeyCode;

struct SnakeTile {
    x: u32,
    y: u32,
}

pub struct Snake {
    compositor: Compositor,
    _root: SpriteVisual,

    game_board: ContainerVisual,
    tiles: Vec<SpriteVisual>,
    selection_visual: SpriteVisual,

    snakes: VecDeque<SnakeTile>,

    game_board_width: i32,
    game_board_height: i32,
    tile_size: Vector2,
    margin: Vector2,
    game_board_margin: Vector2,
    parent_size: Vector2,

    game_over: bool,
    snake_direction: SnakeDirection,
    snake_map: Vec<bool>,
    apple_index: Option<usize>,
}

unsafe impl Send for Snake {}

#[derive(Debug, PartialEq)]
pub enum SnakeDirection {
    UP,
    DOWN,
    RIGHT,
    LEFT,
}

impl Snake {
    pub fn new(parent_visual: &ContainerVisual, parent_size: &Vector2) -> winrt::Result<Self> {
        let compositor = parent_visual.compositor()?;
        let root = compositor.create_sprite_visual()?;

        root.set_relative_size_adjustment(Vector2 { x: 1.0, y: 1.0 })?;
        root.set_brush(compositor.create_color_brush_with_color(Colors::black()?)?)?;
        root.set_border_mode(CompositionBorderMode::Hard)?;
        parent_visual.children()?.insert_at_top(&root)?;

        let tile_size = Vector2 { x: 25.0, y: 25.0 };
        let margin = Vector2 { x: 2.5, y: 2.5 };
        let game_board_margin = Vector2 { x: 100.0, y: 100.0 };

        let game_board = compositor.create_container_visual()?;
        game_board.set_relative_offset_adjustment(Vector3 {
            x: 0.5,
            y: 0.5,
            z: 0.0,
        })?;
        game_board.set_anchor_point(Vector2 { x: 0.5, y: 0.5 })?;
        root.children()?.insert_at_top(&game_board)?;

        let selection_visual = compositor.create_sprite_visual()?;
        let color_brush = compositor.create_color_brush_with_color(Colors::red()?)?;
        let nine_grid_brush = compositor.create_nine_grid_brush()?;
        nine_grid_brush.set_insets_with_values(margin.x, margin.y, margin.x, margin.y)?;
        nine_grid_brush.set_is_center_hollow(true)?;
        nine_grid_brush.set_source(color_brush)?;
        selection_visual.set_brush(nine_grid_brush)?;
        selection_visual.set_offset(Vector3::from_vector2(&margin * -1.0, 0.0))?;
        selection_visual.set_is_visible(false)?;
        selection_visual.set_size(&tile_size + &margin * 2.0)?;
        root.children()?.insert_at_top(&selection_visual)?;

        let mut result = Self {
            compositor: compositor,
            _root: root,

            game_board: game_board,
            tiles: Vec::new(),
            selection_visual: selection_visual,
            snakes: VecDeque::new(),

            game_board_width: 0,
            game_board_height: 0,
            tile_size: tile_size,
            margin: margin,
            game_board_margin: game_board_margin,
            parent_size: parent_size.clone(),

            game_over: false,
            snake_direction: SnakeDirection::RIGHT,
            snake_map: Vec::new(),
            apple_index: None,
        };

        result.new_game(16, 16)?;
        result.create_snake();
        result.generate_apple();
        result.on_parent_size_changed(parent_size)?;

        Ok(result)
    }

    pub fn tick(&mut self) {
        println!("tick");
        match self.move_snake() {
            Ok(()) => {}
            _ => {
                self.new_game(16, 16);
                self.create_snake();
                self.generate_apple();
            }
        }
    }

    fn move_snake(&mut self) -> Result<(), ()> {
        let x: u32;
        let y: u32;
        if self.snakes.len() == 0 {
            return Err(());
        }
        let head = &self.snakes[self.snakes.len() - 1];
        match self.snake_direction {
            SnakeDirection::DOWN => {
                if head.y == self.game_board_height as u32 - 1 {
                    return Err(());
                }
                x = head.x;
                y = head.y + 1;
            }
            SnakeDirection::LEFT => {
                if head.x == 0 {
                    return Err(());
                }
                x = head.x - 1;
                y = head.y;
            }
            SnakeDirection::RIGHT => {
                if head.x == self.game_board_width as u32 - 1 {
                    return Err(());
                }
                x = head.x + 1;
                y = head.y;
            }
            SnakeDirection::UP => {
                if head.y == 0 {
                    return Err(());
                }
                x = head.x;
                y = head.y - 1;
            }
        }

        // Add head :
        match self.new_snake_tile(x, y) {
            Ok(tile) => {
                self.snakes.push_back(tile);
            }
            _ => {
                return Err(());
            }
        }

        if self.is_apple(x, y) {
            self.generate_apple()?;
        } else {
            // Remove tail :
            match &self.snakes.pop_front() {
                Some(tail) => {
                    let index = self.compute_index_from_u32(tail.x, tail.y);
                    match self.set_tile_color_to_blue(index) {
                        Ok(()) => {}
                        _ => {
                            return Err(());
                        }
                    }
                    self.snake_map[index] = false;
                }
                _ => {
                    return Err(());
                }
            }
        }
        return Ok(());
    }

    fn is_apple(&mut self, x: u32, y: u32) -> bool {
        match self.apple_index {
            Some(index) => {
                return index == self.compute_index_from_u32(x, y);
            }
            _ => {
                return false;
            }
        }
    }

    fn generate_apple(&mut self) -> Result<(), ()> {
        let mut available_position: u32 = 0;
        for index in 0..self.game_board_height * self.game_board_width {
            if !self.snake_map[index as usize] {
                available_position = available_position + 1;
            }
        }
        if available_position == 0 {
            // No position available
            return Err(());
        }
        let between = Uniform::from(0..(self.game_board_width) as usize);
        let mut rng = rand::thread_rng();
        let mut x: usize;
        loop {
            x = between.sample(&mut rng);
            let mut do_break: bool = false;
            for y in 0..self.game_board_height as u32 {
                let index: usize = self.compute_index_from_u32(x as u32, y);
                if !self.snake_map[index] {
                    // It's ok.
                    do_break = true;
                    break;
                }
            }
            if do_break {
                break;
            }
        }

        let mut y: usize;
        while {
            y = between.sample(&mut rng);
            let index: usize = self.compute_index_from_u32(x as u32, y as u32);
            self.snake_map[index]
        } {}

        let apple_index: usize = self.compute_index_from_u32(x as u32, y as u32);

        self.apple_index = Some(apple_index);

        match self.set_tile_color_to_red(apple_index) {
            _ => {}
        }
        return Ok(());
    }

    pub fn key_press(&mut self, key: VirtualKeyCode) {
        match key {
            VirtualKeyCode::Z => {
                if self.snake_direction != SnakeDirection::DOWN {
                    self.set_snake_direction(SnakeDirection::UP);
                }
            }
            VirtualKeyCode::Up => {
                if self.snake_direction != SnakeDirection::DOWN {
                    self.set_snake_direction(SnakeDirection::UP);
                }
            }
            VirtualKeyCode::Q => {
                if self.snake_direction != SnakeDirection::RIGHT {
                    self.set_snake_direction(SnakeDirection::LEFT);
                }
            }
            VirtualKeyCode::Left => {
                if self.snake_direction != SnakeDirection::RIGHT {
                    self.set_snake_direction(SnakeDirection::LEFT);
                }
            }
            VirtualKeyCode::S => {
                if self.snake_direction != SnakeDirection::UP {
                    self.set_snake_direction(SnakeDirection::DOWN);
                }
            }
            VirtualKeyCode::Down => {
                if self.snake_direction != SnakeDirection::UP {
                    self.set_snake_direction(SnakeDirection::DOWN);
                }
            }
            VirtualKeyCode::D => {
                if self.snake_direction != SnakeDirection::LEFT {
                    self.set_snake_direction(SnakeDirection::RIGHT);
                }
            }
            VirtualKeyCode::Right => {
                if self.snake_direction != SnakeDirection::LEFT {
                    self.set_snake_direction(SnakeDirection::RIGHT);
                }
            }
            _ => {}
        }
    }

    fn set_snake_direction(&mut self, direction: SnakeDirection) {
        self.snake_direction = direction;
    }

    fn on_parent_size_changed(&mut self, new_size: &Vector2) -> winrt::Result<()> {
        self.parent_size = new_size.clone();
        self.update_board_scale(new_size)?;
        Ok(())
    }

    fn new_snake_tile(&mut self, x: u32, y: u32) -> Result<SnakeTile, ()> {
        let index = self.compute_index_from_u32(x, y);

        if self.snake_map[index] {
            return Err(());
        }

        self.snake_map[index] = true;

        match self.set_tile_color_to_pink(index) {
            Ok(()) => {}
            _ => {
                return Err(());
            }
        }

        let snake_tile = SnakeTile { x: x, y: y };
        return Ok(snake_tile);
    }

    fn set_tile_color_to_blue(&mut self, index: usize) -> winrt::Result<()> {
        let visual = &self.tiles[index];
        visual.set_brush(
            self.compositor
                .create_color_brush_with_color(Colors::blue()?)?,
        )?;
        Ok(())
    }

    fn set_tile_color_to_pink(&mut self, index: usize) -> winrt::Result<()> {
        let visual = &self.tiles[index];
        visual.set_brush(
            self.compositor
                .create_color_brush_with_color(Colors::pink()?)?,
        )?;
        Ok(())
    }

    fn set_tile_color_to_red(&mut self, index: usize) -> winrt::Result<()> {
        let visual = &self.tiles[index];
        visual.set_brush(
            self.compositor
                .create_color_brush_with_color(Colors::red()?)?,
        )?;
        Ok(())
    }

    fn new_game(&mut self, board_width: i32, board_height: i32) -> winrt::Result<()> {
        self.game_board_width = board_width;
        self.game_board_height = board_height;

        self.game_board.children()?.remove_all()?;
        self.tiles.clear();

        self.game_board.set_size(
            (&self.tile_size + &self.margin)
                * Vector2 {
                    x: self.game_board_width as f32,
                    y: self.game_board_height as f32,
                },
        )?;

        for x in 0..self.game_board_width {
            for y in 0..self.game_board_height {
                let visual = self.compositor.create_sprite_visual()?;
                visual.set_size(&self.tile_size)?;
                visual.set_center_point(Vector3::from_vector2(&self.tile_size / 2.0, 0.0))?;
                visual.set_offset(Vector3::from_vector2(
                    (&self.margin / 2.0)
                        + ((&self.tile_size + &self.margin)
                            * Vector2 {
                                x: x as f32,
                                y: y as f32,
                            }),
                    0.0,
                ))?;
                visual.set_brush(
                    self.compositor
                        .create_color_brush_with_color(Colors::blue()?)?,
                )?;

                self.game_board.children()?.insert_at_top(&visual)?;
                self.tiles.push(visual);
            }
        }

        self.snake_map.clear();
        for _x in 0..self.game_board_width {
            for _y in 0..self.game_board_height {
                self.snake_map.push(false);
            }
        }
        self.snakes.clear();

        self.game_over = false;

        self.selection_visual.set_is_visible(false)?;

        self.update_board_scale(&self.parent_size.clone())?;

        self.snake_direction = SnakeDirection::RIGHT;

        Ok(())
    }

    fn create_snake(&mut self) -> Result<(), ()> {
        // Head :
        let mut index: u32 = 0;
        loop {
            let x: u32 = index + self.game_board_width as u32 / 2 - 1;
            let y: u32 = self.game_board_height as u32 / 2;
            let tile = self.new_snake_tile(x, y)?;
            self.snakes.push_back(tile);
            index = index + 1;
            if index == 3 {
                break;
            }
        }
        Ok(())
    }

    fn compute_scale_factor_from_size(&self, window_size: &Vector2) -> winrt::Result<f32> {
        let board_size = self.game_board.size()?;
        let board_size = board_size + &self.game_board_margin;

        let window_ratio = window_size.x / window_size.y;
        let board_ratio = board_size.x / board_size.y;

        let mut scale_factor = window_size.x / board_size.x;
        if window_ratio > board_ratio {
            scale_factor = window_size.y / board_size.y;
        }

        Ok(scale_factor)
    }

    fn update_board_scale(&mut self, window_size: &Vector2) -> winrt::Result<()> {
        let scale_factor = self.compute_scale_factor_from_size(window_size)?;
        self.game_board.set_scale(Vector3 {
            x: scale_factor,
            y: scale_factor,
            z: 1.0,
        })?;
        Ok(())
    }

    fn compute_index_from_u32(&self, x: u32, y: u32) -> usize {
        (x * self.game_board_height as u32 + y) as usize
    }
}
