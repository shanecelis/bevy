//! This example shows how to manually render 2d items using "mid level render apis" with a custom
//! pipeline for 2d meshes.
//! It doesn't use the [`Material2d`] abstraction, but changes the vertex buffer to include vertex color.
//! Check out the "mesh2d" example for simpler / higher level 2d meshes.
//!
//! [`Material2d`]: bevy::sprite::Material2d

use bevy::{
    color::palettes::basic::YELLOW,
    core_pipeline::core_2d::Transparent2d,
    math::FloatOrd,
    prelude::*,
    render::{
        mesh::{GpuMesh, Indices, MeshVertexAttribute},
        render_asset::{RenderAssetUsages, RenderAssets},
        render_phase::{
            AddRenderCommand, DrawFunctions, PhaseItemExtraIndex, SetItemPipeline,
            SortedRenderPhase,
        },
        render_resource::{
            BlendState, ColorTargetState, ColorWrites, Face, FragmentState, FrontFace,
            MultisampleState, PipelineCache, PolygonMode, PrimitiveState, PrimitiveTopology,
            RenderPipelineDescriptor, SpecializedRenderPipeline, SpecializedRenderPipelines,
            TextureFormat, VertexBufferLayout, VertexFormat, VertexState, VertexStepMode,
            binding_types::storage_buffer, *
        },
        renderer::{RenderDevice},
        texture::BevyDefault,
        view::{ExtractedView, ViewTarget, VisibleEntities},
        Extract, Render, RenderApp, RenderSet,
    },
    sprite::{
        extract_mesh2d, DrawMesh2d, Material2dBindGroupId, Mesh2dHandle, Mesh2dPipeline,
        Mesh2dPipelineKey, Mesh2dTransforms, MeshFlags, RenderMesh2dInstance, SetMesh2dBindGroup,
        SetMesh2dViewBindGroup, WithMesh2d,
    },
    utils::EntityHashMap,
};
use std::f32::consts::PI;

fn main() {
    App::new()
        .init_resource::<DistBuffer>()
        .add_plugins((DefaultPlugins, ColoredMesh2dPlugin))
        .add_systems(Startup, star)
        .run();
}

fn calc_dist(mesh: &Mesh) -> Vec<[f32; 3]> {
    assert!(mesh.indices().is_none(), "`insert_dist` can't work on indexed geometry. Consider calling `Mesh::duplicate_vertices`.");

    assert!(
        matches!(mesh.primitive_topology(), PrimitiveTopology::TriangleList),
        "`insert_dist` can only work on `TriangleList`s"
    );

    let positions = mesh
        .attribute(Mesh::ATTRIBUTE_POSITION)
        .unwrap()
        .as_float3()
        .expect("`Mesh::ATTRIBUTE_POSITION` vertex attributes should be of type `float3`");

    let dists: Vec<_> = positions
        .chunks_exact(3)
        .flat_map(|p| tri_dist(p[0], p[1], p[2]))
        .collect();

    dists
}

fn insert_dist(mesh: &mut Mesh) {
    let dists = calc_dist(mesh);
    mesh.insert_attribute(MeshVertexAttribute::new("Tri_Dist", 2, VertexFormat::Float32x3), dists);
}

fn tri_dist(a: [f32; 3], b: [f32; 3], c: [f32; 3]) -> [[f32; 3]; 3] {
    let (p0, p1, p2) = (Vec3::from(a), Vec3::from(b), Vec3::from(c));
    let v0 = p2.xy() - p1.xy();
    let v1 = p2.xy() - p0.xy();
    let v2 = p1.xy() - p0.xy();
    let area = (v1.x * v2.y - v1.y * v2.x).abs();
    let mut d0 = Vec3::X * area/v0.length();
    let mut d1 = Vec3::Y * area/v1.length();
    let mut d2 = Vec3::Z * area/v2.length();
    if v0.length() > v1.length() {
        if v0.length() > v2.length() {
            // v0 is max
            d0 *= 100.0;
        } else {
            // v2 is max
            d2 *= 100.0;
        }
    } else {
        if v1.length() > v2.length() {
            // v1 is max
            d1 *= 100.0;
        } else {
            // v2 is max
            d2 *= 100.0;
        }
    }

    [d0.into(), d1.into(), d2.into()]
}

fn star(
    mut commands: Commands,
    render_device: Res<RenderDevice>,
    // We will add a new Mesh for the star being created
    mut meshes: ResMut<Assets<Mesh>>,
    mut dist_buffer: ResMut<DistBuffer>
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
        RenderAssetUsages::RENDER_WORLD,
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
    // And a RGB color attribute as well
    let mut v_color: Vec<u32> = vec![LinearRgba::BLACK.as_u32()];
    v_color.extend_from_slice(&[LinearRgba::from(YELLOW).as_u32(); 10]);
    star.insert_attribute(
        MeshVertexAttribute::new("Vertex_Color", 1, VertexFormat::Uint32),
        v_color,
    );

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
    insert_dist(&mut star);
    let dist = calc_dist(&star);
    dist_buffer.0 = Some(render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("dist_buffer"),
            contents: bytemuck::cast_slice(dist.as_slice()),
            usage: BufferUsages::STORAGE | BufferUsages::VERTEX,
        }));


    // We can now spawn the entities for the star and the camera
    commands.spawn((
        // We use a marker component to identify the custom colored meshes
        ColoredMesh2d,
        // The `Handle<Mesh>` needs to be wrapped in a `Mesh2dHandle` to use 2d rendering instead of 3d
        Mesh2dHandle(meshes.add(star)),
        // This bundle's components are needed for something to be rendered
        SpatialBundle::INHERITED_IDENTITY,
    ));

    // Spawn the camera
    commands.spawn(Camera2dBundle::default());
}

/// A marker component for colored 2d meshes
#[derive(Component, Default)]
pub struct ColoredMesh2d;

/// Custom pipeline for 2d meshes with vertex colors
#[derive(Resource)]
pub struct ColoredMesh2dPipeline {
    /// this pipeline wraps the standard [`Mesh2dPipeline`]
    mesh2d_pipeline: Mesh2dPipeline,
}

impl FromWorld for ColoredMesh2dPipeline {
    fn from_world(world: &mut World) -> Self {
        Self {
            mesh2d_pipeline: Mesh2dPipeline::from_world(world),
        }
    }
}

// We implement `SpecializedPipeline` to customize the default rendering from `Mesh2dPipeline`
impl SpecializedRenderPipeline for ColoredMesh2dPipeline {
    type Key = Mesh2dPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        // Customize how to store the meshes' vertex attributes in the vertex buffer
        // Our meshes only have position and color
        let formats = [
            // Position
            VertexFormat::Float32x3,
            // Color
            VertexFormat::Uint32,
            // Dist
            VertexFormat::Float32x3,
        ];

        let vertex_layout =
            VertexBufferLayout::from_vertex_formats(VertexStepMode::Vertex, formats);
        let mut dist_layout =
            VertexBufferLayout::from_vertex_formats(VertexStepMode::Vertex, [VertexFormat::Float32x3]);
        dist_layout.attributes[0].shader_location = 2;

        let format = match key.contains(Mesh2dPipelineKey::HDR) {
            true => ViewTarget::TEXTURE_FORMAT_HDR,
            false => TextureFormat::bevy_default(),
        };

        RenderPipelineDescriptor {
            vertex: VertexState {
                // Use our custom shader
                shader: COLORED_MESH2D_SHADER_HANDLE,
                entry_point: "vertex".into(),
                shader_defs: vec![],
                // Use our custom vertex buffer
                buffers: vec![vertex_layout,
                              // dist_layout
                ],
            },
            fragment: Some(FragmentState {
                // Use our custom shader
                shader: COLORED_MESH2D_SHADER_HANDLE,
                shader_defs: vec![],
                entry_point: "fragment".into(),
                targets: vec![Some(ColorTargetState {
                    format,
                    blend: Some(BlendState::ALPHA_BLENDING),
                    write_mask: ColorWrites::ALL,
                })],
            }),
            // Use the two standard uniforms for 2d meshes
            layout: vec![
                // Bind group 0 is the view uniform
                self.mesh2d_pipeline.view_layout.clone(),
                // Bind group 1 is the mesh uniform
                self.mesh2d_pipeline.mesh_layout.clone(),
            ],
            push_constant_ranges: vec![],
            primitive: PrimitiveState {
                front_face: FrontFace::Ccw,
                cull_mode: Some(Face::Back),
                unclipped_depth: false,
                polygon_mode: PolygonMode::Fill,
                conservative: false,
                topology: key.primitive_topology(),
                strip_index_format: None,
            },
            depth_stencil: None,
            multisample: MultisampleState {
                count: key.msaa_samples(),
                mask: !0,
                alpha_to_coverage_enabled: false,
            },
            label: Some("colored_mesh2d_pipeline".into()),
        }
    }
}

use bevy::ecs::system::{lifetimeless::SRes, SystemParamItem};
use bevy::sprite::RenderMesh2dInstances;
use bevy::render::{
    render_phase::{TrackedRenderPass, RenderCommand, RenderCommandResult},
    mesh::GpuBufferInfo,
};

pub struct MyDrawMesh2d;
impl<P: bevy::render::render_phase::PhaseItem> bevy::render::render_phase::RenderCommand<P> for MyDrawMesh2d {
    type Param = (SRes<RenderAssets<GpuMesh>>, SRes<RenderMesh2dInstances>, SRes<DistBuffer>);
    type ViewQuery = ();
    type ItemQuery = ();

    #[inline]
    fn render<'w>(
        item: &P,
        _view: (),
        _item_query: Option<()>,
        (meshes, render_mesh2d_instances, dist_buffer): SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let meshes = meshes.into_inner();
        let render_mesh2d_instances = render_mesh2d_instances.into_inner();
        let dist_buffer = dist_buffer.into_inner();

        let Some(RenderMesh2dInstance { mesh_asset_id, .. }) =
            render_mesh2d_instances.get(&item.entity())
        else {
            return RenderCommandResult::Failure;
        };
        let Some(gpu_mesh) = meshes.get(*mesh_asset_id) else {
            return RenderCommandResult::Failure;
        };

        let Some(ref dist_buffer) = dist_buffer.0 else {
            return RenderCommandResult::Failure;
        };

        pass.set_vertex_buffer(0, gpu_mesh.vertex_buffer.slice(..));
        pass.set_vertex_buffer(1, dist_buffer.slice(..));

        let batch_range = item.batch_range();
        match &gpu_mesh.buffer_info {
            GpuBufferInfo::Indexed {
                buffer,
                index_format,
                count,
            } => {
                pass.set_index_buffer(buffer.slice(..), 0, *index_format);
                pass.draw_indexed(0..*count, 0, batch_range.clone());
            }
            GpuBufferInfo::NonIndexed => {
                pass.draw(0..gpu_mesh.vertex_count, batch_range.clone());
            }
        }
        RenderCommandResult::Success
    }
}

// This specifies how to render a colored 2d mesh
type DrawColoredMesh2d = (
    // Set the pipeline
    SetItemPipeline,
    // Set the view uniform as bind group 0
    SetMesh2dViewBindGroup<0>,
    // Set the mesh uniform as bind group 1
    SetMesh2dBindGroup<1>,
    // Draw the mesh
    // MyDrawMesh2d,
    DrawMesh2d,
);

// The custom shader can be inline like here, included from another file at build time
// using `include_str!()`, or loaded like any other asset with `asset_server.load()`.
const COLORED_MESH2D_SHADER: &str = r"
// Import the standard 2d mesh uniforms and set their bind groups
#import bevy_sprite::mesh2d_functions

// The structure of the vertex buffer is as specified in `specialize()`
struct Vertex {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
    @location(1) color: u32,
    @location(2) dist: vec3<f32>,
};

struct VertexOutput {
    // The vertex shader must set the on-screen position of the vertex
    @builtin(position) clip_position: vec4<f32>,
    // We pass the vertex color to the fragment shader in location 0
    @location(0) color: vec4<f32>,
    @location(1) dist: vec3<f32>,
};

/// Entry point for the vertex shader
@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    var out: VertexOutput;
    // Project the world position of the mesh into screen position
    let model = mesh2d_functions::get_model_matrix(vertex.instance_index);
    out.clip_position = mesh2d_functions::mesh2d_position_local_to_clip(model, vec4<f32>(vertex.position, 1.0));
    // Unpack the `u32` from the vertex buffer into the `vec4<f32>` used by the fragment shader
    out.color = vec4<f32>((vec4<u32>(vertex.color) >> vec4<u32>(0u, 8u, 16u, 24u)) & vec4<u32>(255u)) / 255.0;
    out.dist = vertex.dist;
    return out;
}

// The input of the fragment shader must correspond to the output of the vertex shader for all `location`s
struct FragmentInput {
    // The color is interpolated between vertices by default
    @location(0) color: vec4<f32>,
    @location(1) dist: vec3<f32>,
};
const WIRE_COL: vec4<f32> = vec4(1.0, 0.0, 0.0, 1.0);

/// Entry point for the fragment shader
@fragment
fn fragment(in: FragmentInput) -> @location(0) vec4<f32> {
    let d = min(in.dist[0], min(in.dist[1], in.dist[2]));
    let I = exp2(-2.0 * d * d);
    return I * WIRE_COL + (1.0 - I) * in.color;
}
";

/// Plugin that renders [`ColoredMesh2d`]s
pub struct ColoredMesh2dPlugin;

/// Handle to the custom shader with a unique random ID
pub const COLORED_MESH2D_SHADER_HANDLE: Handle<Shader> =
    Handle::weak_from_u128(13828845428412094821);

/// Our custom pipeline needs its own instance storage
#[derive(Resource, Deref, DerefMut, Default)]
pub struct RenderColoredMesh2dInstances(EntityHashMap<Entity, RenderMesh2dInstance>);

impl Plugin for ColoredMesh2dPlugin {
    fn build(&self, app: &mut App) {
        // Load our custom shader
        let mut shaders = app.world_mut().resource_mut::<Assets<Shader>>();
        shaders.insert(
            &COLORED_MESH2D_SHADER_HANDLE,
            Shader::from_wgsl(COLORED_MESH2D_SHADER, file!()),
        );

        // Register our custom draw function, and add our render systems
        app.get_sub_app_mut(RenderApp)
            .unwrap()
            .add_render_command::<Transparent2d, DrawColoredMesh2d>()
            .init_resource::<SpecializedRenderPipelines<ColoredMesh2dPipeline>>()
            .init_resource::<RenderColoredMesh2dInstances>()
            .add_systems(
                ExtractSchedule,
                extract_colored_mesh2d.after(extract_mesh2d),
            )
            .add_systems(Render, queue_colored_mesh2d.in_set(RenderSet::QueueMeshes));
    }

    fn finish(&self, app: &mut App) {
        // Register our custom pipeline
        app.get_sub_app_mut(RenderApp)
            .unwrap()
            .init_resource::<ColoredMesh2dPipeline>();
    }
}

/// Extract the [`ColoredMesh2d`] marker component into the render app
pub fn extract_colored_mesh2d(
    mut commands: Commands,
    mut previous_len: Local<usize>,
    // When extracting, you must use `Extract` to mark the `SystemParam`s
    // which should be taken from the main world.
    query: Extract<
        Query<(Entity, &ViewVisibility, &GlobalTransform, &Mesh2dHandle), With<ColoredMesh2d>>,
    >,
    mut render_mesh_instances: ResMut<RenderColoredMesh2dInstances>,
) {
    let mut values = Vec::with_capacity(*previous_len);
    for (entity, view_visibility, transform, handle) in &query {
        if !view_visibility.get() {
            continue;
        }

        let transforms = Mesh2dTransforms {
            transform: (&transform.affine()).into(),
            flags: MeshFlags::empty().bits(),
        };

        values.push((entity, ColoredMesh2d));
        render_mesh_instances.insert(
            entity,
            RenderMesh2dInstance {
                mesh_asset_id: handle.0.id(),
                transforms,
                material_bind_group_id: Material2dBindGroupId::default(),
                automatic_batching: false,
            },
        );
    }
    *previous_len = values.len();
    commands.insert_or_spawn_batch(values);
}

/// Queue the 2d meshes marked with [`ColoredMesh2d`] using our custom pipeline and draw function
#[allow(clippy::too_many_arguments)]
pub fn queue_colored_mesh2d(
    transparent_draw_functions: Res<DrawFunctions<Transparent2d>>,
    colored_mesh2d_pipeline: Res<ColoredMesh2dPipeline>,
    mut pipelines: ResMut<SpecializedRenderPipelines<ColoredMesh2dPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    msaa: Res<Msaa>,
    render_meshes: Res<RenderAssets<GpuMesh>>,
    render_mesh_instances: Res<RenderColoredMesh2dInstances>,
    mut views: Query<(
        &VisibleEntities,
        &mut SortedRenderPhase<Transparent2d>,
        &ExtractedView,
    )>,
) {
    if render_mesh_instances.is_empty() {
        return;
    }
    // Iterate each view (a camera is a view)
    for (visible_entities, mut transparent_phase, view) in &mut views {
        let draw_colored_mesh2d = transparent_draw_functions.read().id::<DrawColoredMesh2d>();

        let mesh_key = Mesh2dPipelineKey::from_msaa_samples(msaa.samples())
            | Mesh2dPipelineKey::from_hdr(view.hdr);

        // Queue all entities visible to that view
        for visible_entity in visible_entities.iter::<WithMesh2d>() {
            if let Some(mesh_instance) = render_mesh_instances.get(visible_entity) {
                let mesh2d_handle = mesh_instance.mesh_asset_id;
                let mesh2d_transforms = &mesh_instance.transforms;
                // Get our specialized pipeline
                let mut mesh2d_key = mesh_key;
                if let Some(mesh) = render_meshes.get(mesh2d_handle) {
                    mesh2d_key |=
                        Mesh2dPipelineKey::from_primitive_topology(mesh.primitive_topology());
                }

                let pipeline_id =
                    pipelines.specialize(&pipeline_cache, &colored_mesh2d_pipeline, mesh2d_key);

                let mesh_z = mesh2d_transforms.transform.translation.z;
                transparent_phase.add(Transparent2d {
                    entity: *visible_entity,
                    draw_function: draw_colored_mesh2d,
                    pipeline: pipeline_id,
                    // The 2d render items are sorted according to their z value before rendering,
                    // in order to get correct transparency
                    sort_key: FloatOrd(mesh_z),
                    // This material is not batched
                    batch_range: 0..1,
                    extra_index: PhaseItemExtraIndex::NONE,
                });
            }
        }
    }
}

#[derive(Resource, Default)]
pub struct DistBuffer(Option<Buffer>);

#[derive(Resource)]
struct Buffers {
    // The buffer that will be used by the compute shader
    gpu_buffer: Buffer,
    // The buffer that will be read on the cpu.
    // The `gpu_buffer` will be copied to this buffer every frame
    cpu_buffer: Buffer,
}

#[derive(Resource)]
struct GpuBufferBindGroup(BindGroup);

fn prepare_bind_group(
    mut commands: Commands,
    pipeline: Res<ComputePipeline>,
    render_device: Res<RenderDevice>,
    buffers: Res<Buffers>,
) {
    let bind_group = render_device.create_bind_group(
        None,
        &pipeline.layout,
        &BindGroupEntries::single(buffers.gpu_buffer.as_entire_binding()),
    );
    commands.insert_resource(GpuBufferBindGroup(bind_group));
}

#[derive(Resource)]
struct ComputePipeline {
    layout: BindGroupLayout,
    pipeline: CachedComputePipelineId,
}

impl FromWorld for ComputePipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let layout = render_device.create_bind_group_layout(
            None,
            &BindGroupLayoutEntries::single(
                ShaderStages::COMPUTE,
                storage_buffer::<Vec<u32>>(false),
            ),
        );
        let shader = world.load_asset("shaders/gpu_readback.wgsl");
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("GPU readback compute shader".into()),
            layout: vec![layout.clone()],
            push_constant_ranges: Vec::new(),
            shader: shader.clone(),
            shader_defs: Vec::new(),
            entry_point: "main".into(),
        });
        ComputePipeline { layout, pipeline }
    }
}
