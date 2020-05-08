winrt::import!(
    dependencies
        "os"
    modules
        "windows.foundation.collections"
        "windows.foundation.numerics"
        "windows.ui"
        "windows.ui.composition"
        "windows.ui.composition.desktop"
        "windows.graphics"
        "windows.system"
);

mod interop;
mod numerics;
mod snake;
mod time_span;
mod window_target;

use interop::{create_dispatcher_queue_controller_for_current_thread, ro_initialize, RoInitType};
use snake::Snake;
use std::{
    sync::{Arc, Mutex},
    thread, time,
};
use window_target::CompositionDesktopWindowTargetSource;
use winit::{
    event::{ElementState, Event, WindowEvent},
    event_loop::{ControlFlow, EventLoop},
    window::WindowBuilder,
};

use windows::{foundation::numerics::Vector2, ui::composition::Compositor};

fn run() -> winrt::Result<()> {
    ro_initialize(RoInitType::MultiThreaded)?;
    let _controller = create_dispatcher_queue_controller_for_current_thread()?;

    let event_loop = EventLoop::new();
    let window = WindowBuilder::new().build(&event_loop).unwrap();
    window.set_title("Snake");

    let compositor = Compositor::new()?;
    let target = window.create_window_target(&compositor, false)?;

    let root = compositor.create_container_visual()?;
    root.set_relative_size_adjustment(Vector2 { x: 1.0, y: 1.0 })?;
    target.set_root(&root)?;

    let window_size = window.inner_size();
    let window_size = Vector2 {
        x: window_size.width as f32,
        y: window_size.height as f32,
    };
    let game = Snake::new(&root, &window_size)?;
    let arc = Arc::new(Mutex::new(game));
    {
        let arc = arc.clone();
        thread::spawn(move || loop {
            {
                let mut game = arc.lock().unwrap();
                game.tick();
            }
            thread::sleep(time::Duration::from_millis(100));
        });
    }
    {
        let arc = arc.clone();
        event_loop.run(move |event, _, control_flow| {
            *control_flow = ControlFlow::Wait;
            let mut game = arc.lock().unwrap();
            match event {
                Event::WindowEvent {
                    event: WindowEvent::CloseRequested,
                    window_id,
                } if window_id == window.id() => *control_flow = ControlFlow::Exit,
                Event::WindowEvent {
                    event: WindowEvent::Resized(size),
                    ..
                } => {
                    let size = Vector2 {
                        x: size.width as f32,
                        y: size.height as f32,
                    };
                    game.on_parent_size_changed(&size).unwrap();
                }
                Event::WindowEvent {
                    event: WindowEvent::KeyboardInput { input, .. },
                    ..
                } if input.state == ElementState::Released => match input.virtual_keycode {
                    Some(p) => game.key_press(p),
                    None => println!("No value."),
                },
                _ => (),
            }
        });
    }
}

fn main() {
    let result = run();

    // We do this for nicer HRESULT printing when errors occur.
    if let Err(error) = result {
        error.code().unwrap();
    }
}
