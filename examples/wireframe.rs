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
impl SpecializedRenderPipeline for WireframeMesh2dPipeline {
    type Key = Mesh2dPipelineKey;

    fn specialize(&self, key: Self::Key) -> RenderPipelineDescriptor {
        // Customize how to store the meshes' vertex attributes in the vertex buffer
        let vertex_layout = VertexBufferLayout::from_vertex_formats(
            VertexStepMode::Vertex,
            [
                VertexFormat::Float32x3,
                VertexFormat::Float32x3,
                VertexFormat::Float32x2,
            ],
        );
        let mut dist_layout = VertexBufferLayout::from_vertex_formats(
            VertexStepMode::Vertex,
            [VertexFormat::Float32x4],
        );
        dist_layout.attributes[0].shader_location = 10;

        let format = match key.contains(Mesh2dPipelineKey::HDR) {
            true => ViewTarget::TEXTURE_FORMAT_HDR,
            false => TextureFormat::bevy_default(),
        };

        RenderPipelineDescriptor {
            vertex: VertexState {
                // Use our custom shader
                shader: WIREFRAME_MESH2D_SHADER_HANDLE,
                entry_point: "vertex".into(),
                shader_defs: vec![],
                // Use our custom vertex buffer
                buffers: vec![vertex_layout, dist_layout],
            },
            fragment: Some(FragmentState {
                // Use our custom shader
                shader: WIREFRAME_MESH2D_SHADER_HANDLE,
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
                cull_mode: None, //Some(Face::Back),
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
            label: Some("wireframe_mesh2d_pipeline".into()),
        }
    }
}

pub struct WireframeMesh2dInstance {
    pub render_instance: RenderMesh2dInstance,
    pub dist_buffer: Option<Buffer>,
}

pub struct WireframeDrawMesh2d;
impl<P: PhaseItem> RenderCommand<P> for WireframeDrawMesh2d {
    type Param = (SRes<RenderAssets<GpuMesh>>, SRes<WireframeMesh2dInstances>);
    type ViewQuery = ();
    type ItemQuery = ();

    #[inline]
    fn render<'w>(
        item: &P,
        _view: (),
        _item_query: Option<()>,
        (meshes, wireframe_mesh2d_instances): SystemParamItem<'w, '_, Self::Param>,
        pass: &mut TrackedRenderPass<'w>,
    ) -> RenderCommandResult {
        let meshes = meshes.into_inner();
        let wireframe_mesh2d_instances = wireframe_mesh2d_instances.into_inner();

        let Some(WireframeMesh2dInstance {
            render_instance: RenderMesh2dInstance { mesh_asset_id, .. },
            dist_buffer,
            ..
        }) = wireframe_mesh2d_instances.get(&item.entity())
        else {
            warn!("no instance");
            return RenderCommandResult::Failure;
        };
        let Some(gpu_mesh) = meshes.get(*mesh_asset_id) else {
            warn!("no mesh");
            return RenderCommandResult::Failure;
        };

        pass.set_vertex_buffer(0, gpu_mesh.vertex_buffer.slice(..));
        pass.set_vertex_buffer(1, dist_buffer.as_ref().unwrap().slice(..));

        let batch_range = item.batch_range();
        match &gpu_mesh.buffer_info {
            GpuBufferInfo::Indexed {
                buffer,
                index_format,
                count,
            } => {
                warn!("Tried to draw indexed mesh with wireframe.");
                return RenderCommandResult::Failure;
                // pass.set_index_buffer(buffer.slice(..), 0, *index_format);
                // pass.draw_indexed(0..*count, 0, batch_range.clone());
            }
            GpuBufferInfo::NonIndexed => {
                pass.draw(0..gpu_mesh.vertex_count, batch_range.clone());
                // pass.draw(0..6, batch_range.clone());
            }
        }
        RenderCommandResult::Success
    }
}

// This specifies how to render a colored 2d mesh
type DrawWireframeMesh2d = (
    // Set the pipeline
    SetItemPipeline,
    // Set the view uniform as bind group 0
    SetMesh2dViewBindGroup<0>,
    // Set the mesh uniform as bind group 1
    SetMesh2dBindGroup<1>,
    // Draw the mesh
    WireframeDrawMesh2d,
    // DrawMesh2d,
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
pub struct WireframeMesh2dInstances(EntityHashMap<Entity, WireframeMesh2dInstance>);

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
            .init_resource::<SpecializedRenderPipelines<WireframeMesh2dPipeline>>()
            .init_resource::<WireframeMesh2dInstances>()
            .add_systems(
                ExtractSchedule,
                extract_wireframe_mesh2d.after(extract_mesh2d),
            )
            .add_systems(
                Render,
                prepare_bind_group.in_set(RenderSet::PrepareBindGroups),
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

        values.push((entity, WireframeMesh2d));

        let mesh_asset_id = handle.0.id();
        if !wireframe_mesh_instances.contains_key(&entity) {
            wireframe_mesh_instances.insert(
                entity,
                WireframeMesh2dInstance {
                    render_instance: RenderMesh2dInstance {
                        mesh_asset_id,
                        transforms,
                        material_bind_group_id: Material2dBindGroupId::default(),
                        automatic_batching: false,
                    },
                    dist_buffer: None,
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
    mut pipelines: ResMut<SpecializedRenderPipelines<WireframeMesh2dPipeline>>,
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
            if let Some(wireframe_instance) = wireframe_mesh_instances.get(visible_entity) {
                let mesh_instance = &wireframe_instance.render_instance;
                let mesh2d_handle = mesh_instance.mesh_asset_id;
                let mesh2d_transforms = &mesh_instance.transforms;
                // Get our specialized pipeline
                let mut mesh2d_key = mesh_key;
                if let Some(mesh) = render_meshes.get(mesh2d_handle) {
                    mesh2d_key |=
                        Mesh2dPipelineKey::from_primitive_topology(mesh.primitive_topology());
                    if !matches!(mesh.primitive_topology(), PrimitiveTopology::TriangleList) {
                        panic!(
                            "Expected a TriangleList but got {:?}",
                            mesh.primitive_topology()
                        );
                    }
                }

                let pipeline_id =
                    pipelines.specialize(&pipeline_cache, &wireframe_mesh2d_pipeline, mesh2d_key);

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

fn prepare_bind_group(
    mut commands: Commands,
    pipeline: Res<ScreenspaceDistPipeline>,
    render_device: Res<RenderDevice>,
    pos_buffers: Res<RenderAssets<PosBuffer>>,
    wireframe_mesh: Query<Entity, With<WireframeMesh2d>>,
    mut wireframe_mesh_instances: ResMut<WireframeMesh2dInstances>,
) {
    for entity in wireframe_mesh.iter() {
        let Some(WireframeMesh2dInstance {
            dist_buffer,
            render_instance: RenderMesh2dInstance { mesh_asset_id, .. },
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
        if dist_buffer.is_none() {
            info!(
                "make dist buffer with vertex count {}",
                pos_buffer.vertex_count
            );
            *dist_buffer = Some(render_device.create_buffer(&BufferDescriptor {
                label: Some("dist_buffer"),
                size: (pos_buffer.vertex_count * 4 * 4) as u64,
                usage: BufferUsages::STORAGE | BufferUsages::VERTEX,
                mapped_at_creation: false,
            }))
        }
        let bind_group = render_device.create_bind_group(
            None,
            &pipeline.layout,
            &BindGroupEntries::sequential((
                pos_buffer.buffer.as_entire_buffer_binding(),
                dist_buffer.as_ref().unwrap().as_entire_buffer_binding(),
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
        Some(mesh.count_vertices() * 4 * 4)
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
        dbg!(vertex_count);

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
        let shader = world.load_asset("shaders/screenspace_dist.wgsl");
        let pipeline_cache = world.resource::<PipelineCache>();
        let pipeline = pipeline_cache.queue_compute_pipeline(ComputePipelineDescriptor {
            label: Some("Wireframe compute shader".into()),
            layout: vec![layout.clone()],
            push_constant_ranges: Vec::new(),
            shader: shader.clone(),
            shader_defs: Vec::new(),
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
