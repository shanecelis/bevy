//! Showcases sprite 9 slice scaling and tiling features, enabling usage of
//! sprites in multiple resolutions while keeping it in proportion.
use bevy::{
    image::{ImageLoaderSettings, ImageSampler},
    prelude::*,
};

fn main() {
    App::new()
        .add_plugins(DefaultPlugins)
        .add_systems(Startup, setup)
        .run();
}

fn spawn_sprites(
    commands: &mut Commands,
    texture_handle: Handle<Image>,
    mut position: Vec3,
    slice_border: f32,
    style: TextFont,
    gap: f32,
    original_label: &'static str,
    original_size: Vec2,
    max_corner_scale: f32,
) {
    // Keep the first column 100px wide so rows stay aligned when the original
    // may be much smaller, e.g., 4x4..
    const ORIGINAL_COLUMN: f32 = 100.0;
    let cases = [
        // Reference sprite
        (
            original_label,
            style.clone(),
            original_size,
            ORIGINAL_COLUMN,
            SpriteImageMode::Auto,
        ),
        // Scaled regular sprite
        (
            "Stretched",
            style.clone(),
            Vec2::new(100.0, 200.0),
            100.0,
            SpriteImageMode::Auto,
        ),
        // Stretched Scaled sliced sprite
        (
            "With Slicing",
            style.clone(),
            Vec2::new(100.0, 200.0),
            100.0,
            SpriteImageMode::Sliced(TextureSlicer {
                border: BorderRect::all(slice_border),
                center_scale_mode: SliceScaleMode::Stretch,
                ..default()
            }),
        ),
        // Scaled sliced sprite
        (
            "With Tiling",
            style.clone(),
            Vec2::new(100.0, 200.0),
            100.0,
            SpriteImageMode::Sliced(TextureSlicer {
                border: BorderRect::all(slice_border),
                center_scale_mode: SliceScaleMode::Tile { stretch_value: 0.5 },
                sides_scale_mode: SliceScaleMode::Tile { stretch_value: 0.2 },
                ..default()
            }),
        ),
        // Scaled sliced sprite horizontally
        (
            "With Tiling",
            style.clone(),
            Vec2::new(300.0, 200.0),
            300.0,
            SpriteImageMode::Sliced(TextureSlicer {
                border: BorderRect::all(slice_border),
                center_scale_mode: SliceScaleMode::Tile { stretch_value: 0.2 },
                sides_scale_mode: SliceScaleMode::Tile { stretch_value: 0.3 },
                ..default()
            }),
        ),
        // Scaled sliced sprite horizontally with max scale
        (
            "With Corners Constrained",
            style,
            Vec2::new(300.0, 200.0),
            300.0,
            SpriteImageMode::Sliced(TextureSlicer {
                border: BorderRect::all(slice_border),
                center_scale_mode: SliceScaleMode::Tile { stretch_value: 0.1 },
                sides_scale_mode: SliceScaleMode::Tile { stretch_value: 0.2 },
                max_corner_scale,
            }),
        ),
    ];

    for (label, text_style, size, column_width, scale_mode) in cases {
        position.x += 0.5 * column_width;
        commands.spawn((
            Sprite {
                image: texture_handle.clone(),
                custom_size: Some(size),
                image_mode: scale_mode,
                ..default()
            },
            Transform::from_translation(position),
            children![(
                Text2d::new(label),
                text_style,
                TextLayout::new_with_justify(Justify::Center),
                Transform::from_xyz(0., -0.5 * size.y.max(ORIGINAL_COLUMN) - 10., 0.0),
                bevy::sprite::Anchor::TOP_CENTER,
            )],
        ));
        position.x += 0.5 * column_width + gap;
    }
}

fn setup(mut commands: Commands, asset_server: Res<AssetServer>) {
    commands.spawn((
        Camera2d,
        Projection::Orthographic(OrthographicProjection {
            scale: 1.25,
            ..OrthographicProjection::default_2d()
        }),
    ));

    let font = asset_server.load("fonts/FiraSans-Bold.ttf");
    let style = TextFont {
        font: font.clone(),
        ..default()
    };

    let handle_1 = asset_server.load("textures/slice_square.png");
    let handle_2 = asset_server.load("textures/slice_square_2.png");
    let handle_3 = asset_server
        .load_builder()
        .with_settings(|settings: &mut ImageLoaderSettings| {
            settings.sampler = ImageSampler::nearest();
        })
        .load("textures/slice_square_4_rainbow.png");

    spawn_sprites(
        &mut commands,
        handle_1,
        Vec3::new(-600.0, 250.0, 0.0),
        200.0,
        style.clone(),
        40.,
        "Original",
        Vec2::splat(100.0),
        0.2,
    );

    spawn_sprites(
        &mut commands,
        handle_2,
        Vec3::new(-600.0, 0.0, 0.0),
        80.0,
        style.clone(),
        40.,
        "Original",
        Vec2::splat(100.0),
        0.2,
    );

    spawn_sprites(
        &mut commands,
        handle_3,
        Vec3::new(-600.0, -250.0, 0.0),
        1.0,
        style,
        40.,
        "Original 4x4 (10x)",
        Vec2::splat(40.0),
        0.5,
    );
}
