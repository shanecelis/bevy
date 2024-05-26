use bevy::ecs::system::{
    lifetimeless::{SRes, SResMut, Read},
    SystemParamItem,
};
use bevy::render::{
    mesh::{GpuBufferInfo,MeshVertexBufferLayoutRef},
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

/// A marker component for colored 2d meshes
#[derive(Component, Default)]
pub struct WireframeMesh2d;

/// Custom pipeline for 2d meshes with vertex colors
#[derive(Resource)]
pub struct WireframeMesh2dPipeline {
    /// this pipeline wraps the standard [`Mesh2dPipeline`]
    mesh2d_pipeline: Mesh2dPipeline,
}

impl FromWorld for WireframeMesh2dPipeline {
    fn from_world(world: &mut World) -> Self {
        Self {
            mesh2d_pipeline: Mesh2dPipeline::from_world(world),
        }
    }
}

fn pad(v: [f32; 3]) -> [f32; 4] {
    [v[0], v[1], v[2], 0.0]
}

// We implement `SpecializedPipeline` to customize the default rendering from `Mesh2dPipeline`
impl SpecializedMeshPipeline for WireframeMesh2dPipeline {
    type Key = Mesh2dPipelineKey;

    fn specialize(&self,
                  key: Self::Key,
                  layout: &MeshVertexBufferLayoutRef,
    ) -> Result<RenderPipelineDescriptor, SpecializedMeshPipelineError> {
        let mut descriptor = self.mesh2d_pipeline.specialize(key, layout)?;

        // Customize how to store the meshes' vertex attributes in the vertex buffer
        descriptor.vertex.buffers.push(VertexBufferLayout {
            array_stride: std::mem::size_of::<Vec4>() as u64,
            step_mode: VertexStepMode::Vertex,
            attributes: vec![
                VertexAttribute {
                    format: VertexFormat::Float32x4,
                    offset: 0,
                    shader_location: 10, // shader locations 0-2 are taken up by Position, Normal and UV attributes
                },
            ],
        });

        descriptor.vertex.shader = WIREFRAME_MESH2D_SHADER_HANDLE;
        descriptor.fragment.as_mut().unwrap().shader = WIREFRAME_MESH2D_SHADER_HANDLE;
        descriptor.label = Some("wireframe_mesh2d_pipeline".into());
        Ok(descriptor)
    }
}

pub struct SetDistVertexBuffer<const I: usize>;
impl<P: PhaseItem, const I: usize> RenderCommand<P> for SetDistVertexBuffer<I> {
    type Param = ();
    type ViewQuery = ();
    type ItemQuery = Read<DistBuffer>;

    #[inline]
    fn render<'w>(
        _item: &P,
        _view: (),
        dist_buffer: Option<&'w DistBuffer>,
        mesh2d_bind_group: SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {

        let Some(dist_buffer) = dist_buffer else {
            warn!("no dist");
            return RenderCommandResult::Failure;
        };
        pass.set_vertex_buffer(I, dist_buffer.buffer.slice(..));
        RenderCommandResult::Success
    }
}

// pub struct WireframeDrawMesh2d;
// impl<P: PhaseItem> RenderCommand<P> for WireframeDrawMesh2d {
//     type Param = (SRes<RenderAssets<GpuMesh>>, SRes<WireframeMesh2dInstances>);
//     type ViewQuery = ();
//     type ItemQuery = Read<DistBuffer>;

//     #[inline]
//     fn render<'w>(
//         item: &P,
//         _view: (),
//         dist_buffer: Option<&'w DistBuffer>,
//         (meshes, wireframe_mesh2d_instances): SystemParamItem<'w, '_, Self::Param>,
//         pass: &mut TrackedRenderPass<'w>,
//     ) -> RenderCommandResult {
//         let meshes = meshes.into_inner();
//         let wireframe_mesh2d_instances = wireframe_mesh2d_instances.into_inner();

//         let Some(RenderMesh2dInstance { mesh_asset_id,
//             ..
//         }) = wireframe_mesh2d_instances.get(&item.entity())
//         else {
//             warn!("no instance");
//             return RenderCommandResult::Failure;
//         };
//         let Some(gpu_mesh) = meshes.get(*mesh_asset_id) else {
//             warn!("no mesh");
//             return RenderCommandResult::Failure;
//         };
//         let Some(dist_buffer) = dist_buffer else {
//             warn!("no dist");
//             return RenderCommandResult::Failure;
//         };

//         pass.set_vertex_buffer(0, gpu_mesh.vertex_buffer.slice(..));
//         // pass.set_vertex_buffer(1, dist_buffer.buffer.slice(..));

//         let batch_range = item.batch_range();
//         match &gpu_mesh.buffer_info {
//             GpuBufferInfo::Indexed {
//                 buffer,
//                 index_format,
//                 count,
//             } => {
//                 warn!("Tried to draw indexed mesh with wireframe.");
//                 return RenderCommandResult::Failure;
//                 // pass.set_index_buffer(buffer.slice(..), 0, *index_format);
//                 // pass.draw_indexed(0..*count, 0, batch_range.clone());
//             }
//             GpuBufferInfo::NonIndexed => {
//                 pass.draw(0..gpu_mesh.vertex_count, batch_range.clone());
//                 // pass.draw(0..6, batch_range.clone());
//             }
//         }
//         RenderCommandResult::Success
//     }
// }

// This specifies how to render a colored 2d mesh
type DrawWireframeMesh2d = (
    // Set the pipeline
    SetItemPipeline,
    // Set the view uniform as bind group 0
    SetMesh2dViewBindGroup<0>,
    // Set the mesh uniform as bind group 1
    SetMesh2dBindGroup<1>,
    // Set the dist buffer as vertex buffer 1
    SetDistVertexBuffer<1>,
    // Draw the mesh
    DrawMesh2d,
    // XXX: This was our complicated way of setting the DistVertexBuffer.
    // WireframeDrawMesh2d,
);

// The custom shader can be inline like here, included from another file at build time
// using `include_str!()`, or loaded like any other asset with `asset_server.load()`.
const WIREFRAME_MESH2D_SHADER: &str = r"
// Import the standard 2d mesh uniforms and set their bind groups
#import bevy_sprite::mesh2d_functions

// The structure of the vertex buffer is as specified in `specialize()`
struct Vertex {
    @builtin(instance_index) instance_index: u32,
    @location(0) position: vec3<f32>,
    @location(10) dist: vec4<f32>,
};

struct VertexOutput {
    // The vertex shader must set the on-screen position of the vertex
    @builtin(position) clip_position: vec4<f32>,
    // We pass the vertex color to the fragment shader in location 0
    @location(10) dist: vec4<f32>,
};

/// Entry point for the vertex shader
@vertex
fn vertex(vertex: Vertex) -> VertexOutput {
    var out: VertexOutput;
    // Project the world position of the mesh into screen position
    let model = mesh2d_functions::get_model_matrix(vertex.instance_index);
    out.clip_position = mesh2d_functions::mesh2d_position_local_to_clip(model, vec4<f32>(vertex.position, 1.0));
    // out.clip_position = vec4<f32>(vertex.position / 100.0, 1.0);
    // out.color = vec4<f32>(1.0, 1.0, 0.0, 1.0);
    out.dist = vertex.dist;
    return out;
}

// The input of the fragment shader must correspond to the output of the vertex shader for all `location`s
struct FragmentInput {
    // The color is interpolated between vertices by default
    // @location(0) color: vec4<f32>,
    @location(10) dist: vec4<f32>,
};
const WIRE_COL: vec4<f32> = vec4(1.0, 0.0, 0.0, 1.0);

/// Entry point for the fragment shader
@fragment
fn fragment(in: FragmentInput) -> @location(0) vec4<f32> {
    let color = vec4<f32>(1.0, 1.0, 0.0, 1.0);
    let d = min(in.dist[0], min(in.dist[1], in.dist[2]));
    let I = exp2(-2.0 * d * d);
    // return in.color;
    return I * WIRE_COL + (1.0 - I) * color;
}
";

/// Plugin that renders [`WireframeMesh2d`]s
pub struct WireframeMesh2dPlugin;

/// Handle to the custom shader with a unique random ID
pub const WIREFRAME_MESH2D_SHADER_HANDLE: Handle<Shader> =
    Handle::weak_from_u128(0x1e143d1bcc8b4699859b8863ef474752);

/// Our custom pipeline needs its own instance storage
#[derive(Resource, Deref, DerefMut, Default)]
pub struct WireframeMesh2dInstances(EntityHashMap<Entity, RenderMesh2dInstance>);

impl Plugin for WireframeMesh2dPlugin {
    fn build(&self, app: &mut App) {
        // Load our custom shader
        let mut shaders = app.world_mut().resource_mut::<Assets<Shader>>();
        shaders.insert(
            &WIREFRAME_MESH2D_SHADER_HANDLE,
            Shader::from_wgsl(WIREFRAME_MESH2D_SHADER, file!()),
        );
        app.add_plugins(RenderAssetPlugin::<PosBuffer, GpuImage>::default());

        let render_app = app.sub_app_mut(RenderApp);
        let node = ScreenspaceDistNode::from_world(render_app.world_mut());
        // Register our custom draw function, and add our render systems
        render_app
            .add_render_command::<Transparent2d, DrawWireframeMesh2d>()
            .init_resource::<SpecializedMeshPipelines<WireframeMesh2dPipeline>>()
            .init_resource::<WireframeMesh2dInstances>()
            .add_systems(
                ExtractSchedule,
                extract_wireframe_mesh2d.after(extract_mesh2d),
            )
            .add_systems(
                Render,
                (
                    prepare_dist_buffers.in_set(RenderSet::PrepareResources),
                    prepare_bind_group.in_set(RenderSet::PrepareBindGroups),
                )
            )
            .add_systems(
                Render,
                queue_wireframe_mesh2d.in_set(RenderSet::QueueMeshes),
            );

        let mut render_graph = render_app.world_mut().resource_mut::<RenderGraph>();
        render_graph.add_node(ScreenSpaceDistLabel, node);
        render_graph.add_node_edge(ScreenSpaceDistLabel, bevy::render::graph::CameraDriverLabel);
    }

    fn finish(&self, app: &mut App) {
        // Register our custom pipeline
        app.sub_app_mut(RenderApp)
            .init_resource::<WireframeMesh2dPipeline>()
            .init_resource::<ScreenspaceDistPipeline>();
    }
}

/// Extract the [`WireframeMesh2d`] marker component into the render app
pub fn extract_wireframe_mesh2d(
    mut commands: Commands,
    mut previous_len: Local<usize>,
    // When extracting, you must use `Extract` to mark the `SystemParam`s
    // which should be taken from the main world.
    query: Extract<
        Query<(Entity, &ViewVisibility, &GlobalTransform, &Mesh2dHandle), With<WireframeMesh2d>>,
    >,
    mut wireframe_mesh_instances: ResMut<WireframeMesh2dInstances>,
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

        values.push((entity, (handle.clone(), WireframeMesh2d)));

        let mesh_asset_id = handle.0.id();
        if !wireframe_mesh_instances.contains_key(&entity) {
            wireframe_mesh_instances.insert(
                entity,
                RenderMesh2dInstance {
                    mesh_asset_id,
                    transforms,
                    material_bind_group_id: Material2dBindGroupId::default(),
                    automatic_batching: false,
                },
            );
        }
    }
    *previous_len = values.len();
    commands.insert_or_spawn_batch(values);
}

/// Queue the 2d meshes marked with [`WireframeMesh2d`] using our custom pipeline and draw function
#[allow(clippy::too_many_arguments)]
pub fn queue_wireframe_mesh2d(
    transparent_draw_functions: Res<DrawFunctions<Transparent2d>>,
    wireframe_mesh2d_pipeline: Res<WireframeMesh2dPipeline>,
    mut pipelines: ResMut<SpecializedMeshPipelines<WireframeMesh2dPipeline>>,
    pipeline_cache: Res<PipelineCache>,
    msaa: Res<Msaa>,
    render_meshes: Res<RenderAssets<GpuMesh>>,
    wireframe_mesh_instances: Res<WireframeMesh2dInstances>,
    mut views: Query<(
        &VisibleEntities,
        &mut SortedRenderPhase<Transparent2d>,
        &ExtractedView,
    )>,
) {
    if wireframe_mesh_instances.is_empty() {
        return;
    }
    // Iterate each view (a camera is a view)
    for (visible_entities, mut transparent_phase, view) in &mut views {
        let draw_wireframe_mesh2d = transparent_draw_functions
            .read()
            .id::<DrawWireframeMesh2d>();

        let mesh_key = Mesh2dPipelineKey::from_msaa_samples(msaa.samples())
            | Mesh2dPipelineKey::from_hdr(view.hdr);

        // Queue all entities visible to that view
        for visible_entity in visible_entities.iter::<WithMesh2d>() {
            if let Some(mesh_instance) = wireframe_mesh_instances.get(visible_entity) {
                let mesh2d_handle = mesh_instance.mesh_asset_id;
                let mesh2d_transforms = &mesh_instance.transforms;
                // Get our specialized pipeline
                let mut mesh2d_key = mesh_key;
                let Some(mesh) = render_meshes.get(mesh2d_handle) else {
                    warn!("No mesh");
                    continue;
                };

                mesh2d_key |=
                    Mesh2dPipelineKey::from_primitive_topology(mesh.primitive_topology());
                if !matches!(mesh.primitive_topology(), PrimitiveTopology::TriangleList) {
                    panic!(
                        "Expected a TriangleList but got {:?}",
                        mesh.primitive_topology()
                    );
                }
                let pipeline_id =
                    pipelines.specialize(&pipeline_cache, &wireframe_mesh2d_pipeline, mesh2d_key, &mesh.layout)
                             .expect("specialize 2d pipeline");

                let mesh_z = mesh2d_transforms.transform.translation.z;
                transparent_phase.add(Transparent2d {
                    entity: *visible_entity,
                    draw_function: draw_wireframe_mesh2d,
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

#[derive(Debug, Hash, PartialEq, Eq, Clone, RenderLabel)]
struct ScreenSpaceDistLabel;

#[derive(Component)]
struct WireframeBinding {
    bind_group: BindGroup,
    vertex_count: usize,
}

#[derive(Component)]
pub struct DistBuffer {
    buffer: Buffer,
}

fn prepare_dist_buffers(
    mut commands: Commands,
    meshes: Res<RenderAssets<GpuMesh>>,
    query: Query<(Entity, &Mesh2dHandle)>,
    render_device: Res<RenderDevice>,
) {
    for (entity, handle) in &query {

        let mesh_asset_id = handle.0.id();
        let Some(gpu_mesh) = meshes.get(mesh_asset_id) else {
            warn!("no gpu mesh");
            continue;
        };
        let vertex_count = gpu_mesh.vertex_count as usize;
        let buffer = render_device.create_buffer(&BufferDescriptor {
                label: Some("dist_buffer"),
                size: (std::mem::size_of::<Vec4>() * vertex_count) as u64,
                usage: BufferUsages::STORAGE | BufferUsages::VERTEX,
                mapped_at_creation: false,
            });
        commands.entity(entity).insert(DistBuffer {
            buffer,
        });
    }
}

fn prepare_bind_group(
    mut commands: Commands,
    pipeline: Res<ScreenspaceDistPipeline>,
    render_device: Res<RenderDevice>,
    pos_buffers: Res<RenderAssets<PosBuffer>>,
    wireframe_mesh: Query<(Entity, &DistBuffer), With<WireframeMesh2d>>,
    mut wireframe_mesh_instances: ResMut<WireframeMesh2dInstances>,
) {
    for (entity, dist_buffer) in wireframe_mesh.iter() {
        let Some(RenderMesh2dInstance { mesh_asset_id,
            ..
        }) = wireframe_mesh_instances.get_mut(&entity)
        else {
            warn!("no wireframe mesh 2d");
            return;
        };
        let Some(pos_buffer) = pos_buffers.get(*mesh_asset_id) else {
            warn!("no pos buffer");
            return;
        };
        let bind_group = render_device.create_bind_group(
            None,
            &pipeline.layout,
            &BindGroupEntries::sequential((
                pos_buffer.buffer.as_entire_buffer_binding(),
                dist_buffer.buffer.as_entire_buffer_binding(),
            )),
        );
        let vertex_count = pos_buffer.vertex_count;
        commands.entity(entity).insert(WireframeBinding {
            bind_group,
            vertex_count,
        });
    }
}

struct PosBuffer {
    buffer: Buffer,
    vertex_count: usize,
}

impl RenderAsset for PosBuffer {
    type SourceAsset = Mesh;
    type Param = (SRes<RenderDevice>,);

    #[inline]
    fn asset_usage(mesh: &Self::SourceAsset) -> RenderAssetUsages {
        mesh.asset_usage
    }

    fn byte_len(mesh: &Self::SourceAsset) -> Option<usize> {
        Some(mesh.count_vertices() * std::mem::size_of::<Vec4>())
    }

    fn prepare_asset(
        mesh: Self::SourceAsset,
        (render_device,): &mut SystemParamItem<Self::Param>,
    ) -> Result<Self, PrepareAssetError<Self::SourceAsset>> {
        let Some(VertexAttributeValues::Float32x3(positions)) =
            mesh.attribute(Mesh::ATTRIBUTE_POSITION)
        else {
            warn!("no position vertices");
            return Err(PrepareAssetError::RetryNextUpdate(mesh));
        };
        let v_pos_4: Vec<[f32; 4]> = positions.into_iter().map(|x| pad(*x)).collect();

        let vertex_count = mesh.count_vertices();

        let pos_buffer = render_device.create_buffer_with_data(&BufferInitDescriptor {
            label: Some("pos_buffer"),
            contents: bytemuck::cast_slice(v_pos_4.as_slice()),
            usage: BufferUsages::STORAGE,
        });

        Ok(PosBuffer {
            vertex_count,
            buffer: pos_buffer,
        })
    }
}

#[derive(Resource)]
struct ScreenspaceDistPipeline {
    layout: BindGroupLayout,
    pipeline: CachedComputePipelineId,
}

impl FromWorld for ScreenspaceDistPipeline {
    fn from_world(world: &mut World) -> Self {
        let render_device = world.resource::<RenderDevice>();
        let layout = render_device.create_bind_group_layout(
            "ScreenSpaceDist",
            &BindGroupLayoutEntries::sequential(
                ShaderStages::COMPUTE,
                (
                    storage_buffer_read_only::<Vec<Vec4>>(false),
                    storage_buffer::<Vec<Vec4>>(false),
                ),
            ),
        );
        let shader_defs = vec!["MODEL_DIST".into()];
        let shader = world.load_asset("shaders/screenspace_dist.wgsl");
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("Wireframe compute shader".into()),
            layout: vec![layout.clone()],
            push_constant_ranges: Vec::new(),
            shader,
            shader_defs,
            entry_point: "main".into(),
        });
        ScreenspaceDistPipeline { layout, pipeline }
    }
}

struct ScreenspaceDistNode {
    query: QueryState<&'static WireframeBinding>,
}

impl FromWorld for ScreenspaceDistNode {
    fn from_world(world: &mut World) -> Self {
        Self {
            query: QueryState::new(world),
        }
    }
}

impl render_graph::Node for ScreenspaceDistNode {
    fn update(&mut self, world: &mut World) {
        self.query.update_archetypes(world);
    }

    fn run(
        &self,
        _graph: &mut render_graph::RenderGraphContext,
        render_context: &mut RenderContext,
        world: &World,
    ) -> Result<(), render_graph::NodeRunError> {
        for wireframe_binding in self.query.iter_manual(world) {
            let bind_group = &wireframe_binding.bind_group;
            let pipeline_cache = world.resource::<PipelineCache>();
            let pipeline = world.resource::<ScreenspaceDistPipeline>();

            let mut pass = render_context
                .command_encoder()
                .begin_compute_pass(&ComputePassDescriptor::default());

            let update_pipeline = pipeline_cache
                .get_compute_pipeline(pipeline.pipeline)
                .unwrap();
            pass.set_bind_group(0, &bind_group, &[]);
            pass.set_pipeline(update_pipeline);
            pass.dispatch_workgroups((wireframe_binding.vertex_count / 3) as u32, 1, 1);
        }

        Ok(())
    }
}
