//! This example shows how to manually render 2d items using "mid level render apis" with a custom
//! pipeline for 2d meshes.
//! It doesn't use the [`Material2d`] abstraction, but changes the vertex buffer to include vertex color.
//! Check out the "mesh2d" example for simpler / higher level 2d meshes.
//!
//! [`Material2d`]: bevy::sprite::Material2d

use bevy::ecs::system::{
    lifetimeless::{SRes, SResMut},
    SystemParamItem,
};
use bevy::render::{
    mesh::GpuBufferInfo,
    render_phase::{RenderCommandResult, TrackedRenderPass},
};
use bevy::{
    color::palettes::basic::YELLOW,
    core_pipeline::core_2d::Transparent2d,
    math::FloatOrd,
    prelude::*,
    render::{
        extract_resource::{ExtractResource, ExtractResourcePlugin},
        mesh::{GpuMesh, Indices, MeshVertexAttribute, VertexAttributeValues},
        render_asset::{PrepareAssetError, RenderAssetUsages, RenderAssets},
        render_asset::{RenderAsset, RenderAssetPlugin},
        render_graph::{self, RenderGraph, RenderLabel},
        render_phase::{
            AddRenderCommand, DrawFunctions, PhaseItem, PhaseItemExtraIndex, RenderCommand,
            SetItemPipeline, SortedRenderPhase,
        },
        render_resource::{
            binding_types::{storage_buffer, storage_buffer_read_only},
            BlendState, ColorTargetState, ColorWrites, Face, FragmentState, FrontFace,
            MultisampleState, PipelineCache, PolygonMode, PrimitiveState, PrimitiveTopology,
            RenderPipelineDescriptor, SpecializedRenderPipeline, SpecializedRenderPipelines,
            TextureFormat, VertexBufferLayout, VertexFormat, VertexState, VertexStepMode, *,
        },
        renderer::{RenderContext, RenderDevice},
        texture::{BevyDefault, GpuImage},
        view::{ExtractedView, ViewTarget, VisibleEntities},
        Extract, Render, RenderApp, RenderSet,
    },
    sprite::{
        extract_mesh2d, DrawMesh2d, Material2dBindGroupId, MaterialMesh2dBundle, Mesh2dHandle,
        Mesh2dPipeline, Mesh2dPipelineKey, Mesh2dTransforms, MeshFlags, RenderMesh2dInstance,
        SetMesh2dBindGroup, SetMesh2dViewBindGroup, WithMesh2d,
    },
    utils::EntityHashMap,
};

use bevy::log::LogPlugin;
use std::collections::HashMap;
use std::f32::consts::PI;

#[path = "../wireframe.rs"]
mod wireframe;

use wireframe::*;

fn main() {
    App::new()
        .add_plugins(LogPlugin::default())
        .add_plugins((
            DefaultPlugins.build().disable::<LogPlugin>(),
            WireframeMesh2dPlugin,
        ))
        .add_systems(Startup, star)
        .run();
}

fn star(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    // We will add a new Mesh for the star being created
    mut meshes: ResMut<Assets<Mesh>>,
    mut materials: ResMut<Assets<ColorMaterial>>,
    // mut dist_buffer: ResMut<Buffers>
) {
    // Let's define the mesh for the object we want to draw: a nice star.
    // We will specify here what kind of topology is used to define the mesh,
    // that is, how triangles are built from the vertices. We will use a
    // triangle list, meaning that each vertex of the triangle has to be
    // specified. We set `RenderAssetUsages::RENDER_WORLD`, meaning this mesh
    // will not be accessible in future frames from the `meshes` resource, in
    // order to save on memory once it has been uploaded to the GPU.
    let mut star = Mesh::new(
        PrimitiveTopology::TriangleList,
        // FIXME: Main world is required in order to allow PosBuffer to process
        // the mesh too.
        RenderAssetUsages::RENDER_WORLD | RenderAssetUsages::MAIN_WORLD,
    );

    // Vertices need to have a position attribute. We will use the following
    // vertices (I hope you can spot the star in the schema).
    //
    //        1
    //
    //     10   2
    // 9      0      3
    //     8     4
    //        6
    //   7        5
    //
    // These vertices are specified in 3D space.
    let mut v_pos = vec![[0.0, 0.0, 0.0]];
    for i in 0..10 {
        // The angle between each vertex is 1/10 of a full rotation.
        let a = i as f32 * PI / 5.0;
        // The radius of inner vertices (even indices) is 100. For outer vertices (odd indices) it's 200.
        let r = (1 - i % 2) as f32 * 100.0 + 100.0;
        // Add the vertex position.
        v_pos.push([r * a.sin(), r * a.cos(), 0.0]);
    }

    // Set the position attribute
    star.insert_attribute(Mesh::ATTRIBUTE_POSITION, v_pos);
    // star.insert_attribute(Mesh::ATTRIBUTE_NORMAL, vec![[0.0, 0., 1.]; 10]);
    // star.insert_attribute(Mesh::ATTRIBUTE_UV_0, vec![[0.0, 0.]; 10]);
    // And a RGB color attribute as well
    // let mut v_color: Vec<u32> = vec![LinearRgba::BLACK.as_u32()];
    // v_color.extend_from_slice(&[LinearRgba::from(YELLOW).as_u32(); 10]);
    // star.insert_attribute(
    //     MeshVertexAttribute::new("Vertex_Color", 1, VertexFormat::Uint32),
    //     v_color,
    // );

    // Now, we specify the indices of the vertex that are going to compose the
    // triangles in our star. Vertices in triangles have to be specified in CCW
    // winding (that will be the front face, colored). Since we are using
    // triangle list, we will specify each triangle as 3 vertices
    //   First triangle: 0, 2, 1
    //   Second triangle: 0, 3, 2
    //   Third triangle: 0, 4, 3
    //   etc
    //   Last triangle: 0, 1, 10
    let mut indices = vec![0, 1, 10];
    for i in 2..=10 {
        indices.extend_from_slice(&[0, i, i - 1]);
    }
    star.insert_indices(Indices::U32(indices));
    star.duplicate_vertices();

    // The `Handle<Mesh>` needs to be wrapped in a `Mesh2dHandle` to use 2d
    // rendering instead of 3d.
    let handle = Mesh2dHandle(meshes.add(star));
    commands.spawn((
        WireframeMesh2d,
        handle.clone(),
        SpatialBundle::INHERITED_IDENTITY,
    ));

    // commands.spawn((
    //     WireframeMesh2d,
    //     handle,
    //     SpatialBundle::from_transform(Transform::from_xyz(300.0, 100.0, 1.0)),
    // ));

    let mut shape = Circle { radius: 50.0 };
    // let mut shape = Triangle2d::new(Vec2::new(0.0, 0.0), Vec2::new(0.0, -100.0), Vec2::new(-100.0, -100.0));
    // let mut shape = Rectangle::new(25.0, 50.0);
    // dbg!(shape.winding_order());
    // shape.reverse();
    // dbg!(shape.winding_order());
    let mut circle: Mesh = shape.into();
    // let mut circle: Mesh = Rectangle::new(25.0, 50.0);
    dbg!(circle.count_vertices());
    circle.duplicate_vertices();
    // dbg!(&circle);
    circle.asset_usage = RenderAssetUsages::RENDER_WORLD | RenderAssetUsages::MAIN_WORLD;
    dbg!(circle.count_vertices());
    let handle = Mesh2dHandle(meshes.add(circle.clone()));

    commands.spawn(MaterialMesh2dBundle {
        mesh: handle.clone(),
        material: materials.add(Color::hsl(180.0, 0.95, 0.7)),
        transform: Transform::from_xyz(-300.0, 100.0, 10.0),
        ..default()
    });

    commands.spawn((
        WireframeMesh2d,
        handle,
        // SpatialBundle::INHERITED_IDENTITY,
        SpatialBundle::from_transform(Transform::from_xyz(-300.0, -100.0, 2.0))
    ));

    // Spawn the camera
    commands.spawn(Camera2dBundle::default());
}
