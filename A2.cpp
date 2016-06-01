#include "A2.hpp"
#include "cs488-framework/GlErrorCheck.hpp"

#include <iostream>
using namespace std;

#include <imgui/imgui.h>

#include <glm/gtc/matrix_transform.hpp>
#include <glm/gtc/type_ptr.hpp>
#include <glm/gtx/io.hpp>
using namespace glm;


static const float MAX_SCALE = 5.0f;
static const float MIN_SCALE = 0.5f;
static const float MAX_FOV = glm::radians(160.0f);
static const float MIN_FOV = glm::radians(5.0f);

//----------------------------------------------------------------------------------------
// Constructor
VertexData::VertexData()
	: numVertices(0),
	  index(0)
{
	positions.reserve(kMaxVertices);
	colours.reserve(kMaxVertices);
}


//----------------------------------------------------------------------------------------
// Constructor
A2::A2()
	: m_currentLineColour(vec3(0.0f)),
	  left_mouse_dragging(false),
	  middle_mouse_dragging(false),
	  right_mouse_dragging(false),
	  prev_x_pos(-1),
	  prev_y_pos(-1),
	  curr_mode(ROTATE_MODEL)
{
	VertexData m_vertexData;
}

//----------------------------------------------------------------------------------------
// Destructor
A2::~A2()
{

}

//----------------------------------------------------------------------------------------
/*
 * Called once, at program start.
 */
void A2::init()
{
	glHint(GL_LINE_SMOOTH_HINT, GL_NICEST);
	glEnable(GL_LINE_SMOOTH);


	// Set the background colour.
	glClearColor(0.3, 0.5, 0.7, 1.0);

	createShaderProgram();

	glGenVertexArrays(1, &m_vao);

	enableVertexAttribIndices();

	generateVertexBuffers();

	mapVboDataToVertexAttributeLocation();

	reset();
}

//----------------------------------------------------------------------------------------
void A2::createShaderProgram()
{
	m_shader.generateProgramObject();
	m_shader.attachVertexShader( getAssetFilePath("VertexShader.vs").c_str() );
	m_shader.attachFragmentShader( getAssetFilePath("FragmentShader.fs").c_str() );
	m_shader.link();
}

//----------------------------------------------------------------------------------------
void A2::enableVertexAttribIndices()
{
	glBindVertexArray(m_vao);

	// Enable the attribute index location for "position" when rendering.
	GLint positionAttribLocation = m_shader.getAttribLocation( "position" );
	glEnableVertexAttribArray(positionAttribLocation);

	// Enable the attribute index location for "colour" when rendering.
	GLint colourAttribLocation = m_shader.getAttribLocation( "colour" );
	glEnableVertexAttribArray(colourAttribLocation);

	// Restore defaults
	glBindVertexArray(0);

	CHECK_GL_ERRORS;
}

//----------------------------------------------------------------------------------------
void A2::generateVertexBuffers()
{
	// Generate a vertex buffer to store line vertex positions
	{
		glGenBuffers(1, &m_vbo_positions);

		glBindBuffer(GL_ARRAY_BUFFER, m_vbo_positions);

		// Set to GL_DYNAMIC_DRAW because the data store will be modified frequently.
		glBufferData(GL_ARRAY_BUFFER, sizeof(vec2) * kMaxVertices, nullptr,
				GL_DYNAMIC_DRAW);


		// Unbind the target GL_ARRAY_BUFFER, now that we are finished using it.
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		CHECK_GL_ERRORS;
	}

	// Generate a vertex buffer to store line colors
	{
		glGenBuffers(1, &m_vbo_colours);

		glBindBuffer(GL_ARRAY_BUFFER, m_vbo_colours);

		// Set to GL_DYNAMIC_DRAW because the data store will be modified frequently.
		glBufferData(GL_ARRAY_BUFFER, sizeof(vec3) * kMaxVertices, nullptr,
				GL_DYNAMIC_DRAW);


		// Unbind the target GL_ARRAY_BUFFER, now that we are finished using it.
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		CHECK_GL_ERRORS;
	}
}

//----------------------------------------------------------------------------------------
void A2::mapVboDataToVertexAttributeLocation()
{
	// Bind VAO in order to record the data mapping.
	glBindVertexArray(m_vao);

	// Tell GL how to map data from the vertex buffer "m_vbo_positions" into the
	// "position" vertex attribute index for any bound shader program.
	glBindBuffer(GL_ARRAY_BUFFER, m_vbo_positions);
	GLint positionAttribLocation = m_shader.getAttribLocation( "position" );
	glVertexAttribPointer(positionAttribLocation, 2, GL_FLOAT, GL_FALSE, 0, nullptr);

	// Tell GL how to map data from the vertex buffer "m_vbo_colours" into the
	// "colour" vertex attribute index for any bound shader program.
	glBindBuffer(GL_ARRAY_BUFFER, m_vbo_colours);
	GLint colorAttribLocation = m_shader.getAttribLocation( "colour" );
	glVertexAttribPointer(colorAttribLocation, 3, GL_FLOAT, GL_FALSE, 0, nullptr);

	//-- Unbind target, and restore default values:
	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	CHECK_GL_ERRORS;
}

//---------------------------------------------------------------------------------------
void A2::initLineData()
{
	m_vertexData.numVertices = 0;
	m_vertexData.index = 0;
}

//---------------------------------------------------------------------------------------
void A2::setLineColour (
		const glm::vec3 & colour
) {
	m_currentLineColour = colour;
}

//---------------------------------------------------------------------------------------
void A2::drawLine(
		const glm::vec2 & v0,   // Line Start (NDC coordinate)
		const glm::vec2 & v1    // Line End (NDC coordinate)
) {

	m_vertexData.positions[m_vertexData.index] = v0;
	m_vertexData.colours[m_vertexData.index] = m_currentLineColour;
	++m_vertexData.index;
	m_vertexData.positions[m_vertexData.index] = v1;
	m_vertexData.colours[m_vertexData.index] = m_currentLineColour;
	++m_vertexData.index;

	m_vertexData.numVertices += 2;
}

//----------------------------------------------------------------------------------------
/*
 * Called once per frame, before guiLogic().
 */
void A2::appLogic()
{
	// Place per frame, application logic here ...

	// Call at the beginning of frame, before drawing lines:
	initLineData();

	// Lines for gnomons
	vec4 gnomons[] = {
		vec4(0.0f, 0.0f, 0.0f, 1.0f),
		vec4(1.0f, 0.0f, 0.0f, 1.0f),

		vec4(0.0f, 0.0f, 0.0f, 1.0f),
		vec4(0.0f, 1.0f, 0.0f, 1.0f),

		vec4(0.0f, 0.0f, 0.0f, 1.0f),
		vec4(0.0f, 0.0f, 1.0f, 1.0f),
	};

	// Lines for a cube
	vec4 cube_verts[] = {
		vec4(1.0f, 1.0f, 1.0f, 1.0f),
		vec4(-1.0f, 1.0f, 1.0f, 1.0f),

		vec4(1.0f, 1.0f, 1.0f, 1.0f),
		vec4(1.0f, -1.0f, 1.0f, 1.0f),

		vec4(1.0f, 1.0f, 1.0f, 1.0f),
		vec4(1.0f, 1.0f, -1.0f, 1.0f),
		
		vec4(-1.0f, 1.0f, 1.0f, 1.0f),
		vec4(-1.0f, -1.0f, 1.0f, 1.0f),

		vec4(-1.0f, 1.0f, 1.0f, 1.0f),
		vec4(-1.0f, 1.0f, -1.0f, 1.0f),

		vec4(-1.0f, -1.0f, 1.0f, 1.0f),
		vec4(1.0f, -1.0f, 1.0f, 1.0f),

		vec4(1.0f, -1.0f, 1.0f, 1.0f),
		vec4(1.0f, -1.0f, -1.0f, 1.0f),

		vec4(1.0f, 1.0f, -1.0f, 1.0f),
		vec4(-1.0f, 1.0f, -1.0f, 1.0f),
		
		vec4(1.0f, 1.0f, -1.0f, 1.0f),
		vec4(1.0f, -1.0f, -1.0f, 1.0f),

		vec4(-1.0f, -1.0f, -1.0f, 1.0f),
		vec4(-1.0f, -1.0f, 1.0f, 1.0f),
		
		vec4(-1.0f, -1.0f, -1.0f, 1.0f),
		vec4(-1.0f, 1.0f, -1.0f, 1.0f),

		vec4(-1.0f, -1.0f, -1.0f, 1.0f),
		vec4(1.0f, -1.0f, -1.0f, 1.0f)
	};

	setLineColour(vec3(1.0f, 1.0f, 0.8f));
	int i = 0;
	for (int idx = 0; idx < 12; idx++) {
		vec4 start_3d = proj * view * model_transl_rot * model_scale * cube_verts[i];
		vec4 end_3d = proj * view * model_transl_rot * model_scale * cube_verts[i+1];

		vector<vec4> clipped_pts = clip(start_3d, end_3d);
		if (clipped_pts.size() == 2) {
			vec3 start = vec3(clipped_pts[0].x / clipped_pts[0].w, clipped_pts[0].y / clipped_pts[0].w, 1.0f);
			vec3 end = vec3(clipped_pts[1].x / clipped_pts[1].w, clipped_pts[1].y / clipped_pts[1].w, 1.0f);
			if (curr_mode == VIEWPORT) {
				start = viewport * start;
				end = viewport * end;
			}
			drawLine(
				vec2(start.x, start.y),
				vec2(end.x, end.y)
			);
		}
		i += 2;
	}

	i = 0;
	for (int idx = 0; idx < 3; idx++) {
		switch (idx) {
			case 0:
				setLineColour(vec3(1.0f, 0.0f, 0.0f));
				break;
			case 1:
				setLineColour(vec3(0.0f, 1.0f, 0.0f));
				break;
			case 2:
				setLineColour(vec3(0.0f, 0.0f, 1.0f));
				break;
		}

		// Model gnomons
		vec4 start_3d = proj * view * model_transl_rot * gnomons[i];
		vec4 end_3d = proj * view * model_transl_rot * gnomons[i + 1];

		vector<vec4> clipped_pts = clip(start_3d, end_3d);
		if (clipped_pts.size() == 2) {
			vec3 start = vec3(clipped_pts[0].x / clipped_pts[0].w, clipped_pts[0].y / clipped_pts[0].w, 1.0f);
			vec3 end = vec3(clipped_pts[1].x / clipped_pts[1].w, clipped_pts[1].y / clipped_pts[1].w, 1.0f);
			if (curr_mode == VIEWPORT) {
				start = viewport * start;
				end = viewport * end;
			}
			drawLine(
				vec2(start.x, start.y),
				vec2(end.x, end.y)
			);
		}
		// World gnomons
		start_3d = proj * view * gnomons[i];
		end_3d = proj * view * gnomons[i + 1];

		clipped_pts = clip(start_3d, end_3d);
		if (clipped_pts.size() == 2) {
			vec3 start = vec3(clipped_pts[0].x / clipped_pts[0].w, clipped_pts[0].y / clipped_pts[0].w, 1.0f);
			vec3 end = vec3(clipped_pts[1].x / clipped_pts[1].w, clipped_pts[1].y / clipped_pts[1].w, 1.0f);
			if (curr_mode == VIEWPORT) {
				start = viewport * start;
				end = viewport * end;
			}
			drawLine(
				vec2(start.x, start.y),
				vec2(end.x, end.y)
			);
		}
		i += 2;
	}

	// Draw viewport border
	if (curr_mode == VIEWPORT) {
		vec3 border[] = {
			vec3(-1.0f, -1.0f, 1.0f),
			vec3(1.0f, -1.0f, 1.0f),

			vec3(-1.0f, -1.0f, 1.0f),
			vec3(-1.0f, 1.0f, 1.0f),

			vec3(1.0f, 1.0f, 1.0f),
			vec3(1.0f, -1.0f, 1.0f),

			vec3(1.0f, 1.0f, 1.0f),
			vec3(-1.0f, 1.0f, 1.0f),
		};
		setLineColour(vec3(0.0f, 0.0f, 0.0f));
		int i = 0;
		for (int idx = 0; idx < 4; idx++) {
			vec3 start = viewport * border[i];
			vec3 end = viewport * border[i+1];
			drawLine(
				vec2(start.x, start.y),
				vec2(end.x, end.y)
			);
			i += 2;
		}
	}
}

void A2::reset() {
	// Set up initial matrices
	model_scale = glm::mat4();
	model_transl_rot = glm::mat4();
	view = glm::lookAt(
		glm::vec3(0.0f, 10.0f, 10.0f),
		glm::vec3(0.0f, 0.0f, 0.0f),
		glm::vec3(0.0f, 1.0f, 0.0f));
	fov = glm::radians(30.0f);
	n = 1.0f;
	f = 20.0f;
	v_x_start = 0.05f * m_framebufferWidth;
	v_x_end = 0.95f * m_framebufferWidth;
	v_y_start = 0.05f * m_framebufferHeight;
	v_y_end = 0.95f * m_framebufferHeight;
	updateProjMatrix();
	updateViewportMatrix();
	curr_mode = ROTATE_MODEL;
}

void A2::updateProjMatrix() {
	proj = glm::mat4(
		(1 / tan(fov / 2) / (m_framebufferWidth / m_framebufferHeight)), 0.0f, 0.0f, 0.0f,
		0.0f, 1 / tan(fov / 2), 0.0f, 0.0f,
		0.0f, 0.0f, -(f + n) / (f - n), -1.0f,
		0.0f, 0.0f, -2 * f * n / (f - n), 0.0f);
}

void A2::updateViewportMatrix() {
	float v_x_min = float(glm::min(v_x_start, v_x_end));
	float v_x_max = float(glm::max(v_x_start, v_x_end));
	float v_y_min = float(glm::min(v_y_start, v_y_end));
	float v_y_max = float(glm::max(v_y_start, v_y_end));
	float x_ratio = (v_x_max - v_x_min) / m_framebufferWidth;
	float y_ratio = (v_y_max - v_y_min) / m_framebufferHeight;
	viewport = glm::mat3(
		x_ratio, 0.0f, 0.0f,
		0.0f, y_ratio, 0.0f,
		2.0f * (v_x_min/m_framebufferWidth) - (1 - x_ratio), -2.0f * (v_y_min / m_framebufferHeight) + (1 - y_ratio), 1.0f
	);
}

void A2::rotateViewBy(float degree)
{
	if (left_mouse_dragging) {
		// Rotate around x-axis
		glm::mat4 rotation = glm::mat4(
			1.0f, 0.0f, 0.0f, 0.0f,
			0.0f, cos(degree), sin(degree), 0.0f,
			0.0f, -sin(degree), cos(degree), 0.0f,
			0.0f, 0.0f, 0.0f, 1.0f
		);
		view = glm::inverse(rotation) * view;
	}
	if (middle_mouse_dragging) {
		// Rotate around y-axis
		glm::mat4 rotation = glm::mat4(
			cos(degree), 0.0f, -sin(degree), 0.0f,
			0.0f, 1.0f, 0.0f, 0.0f,
			sin(degree), 0.0f, cos(degree), 0.0f,
			0.0f, 0.0f, 0.0f, 1.0f
		);
		view = glm::inverse(rotation) * view;
	}
	if (right_mouse_dragging) {
		// Rotate around z-axis
		glm::mat4 rotation = glm::mat4(
			cos(degree), sin(degree), 0.0f, 0.0f,
			-sin(degree), cos(degree), 0.0f, 0.0f,
			0.0f, 0.0f, 1.0f, 0.0f,
			0.0f, 0.0f, 0.0f, 1.0f
		);
		view = glm::inverse(rotation) * view;
	}
}

void A2::translateViewBy(float displacement)
{
	if (left_mouse_dragging) {
		// Translate on x-axis
		glm::mat4 translation = glm::mat4(
			1.0f, 0.0f, 0.0f, 0.0f,
			0.0f, 1.0f, 0.0f, 0.0f,
			0.0f, 0.0f, 1.0f, 0.0f,
			displacement, 0.0f, 0.0f, 1.0f
		);
		view = glm::inverse(translation) * view;
	}
	if (middle_mouse_dragging) {
		// Translate on y-axis
		glm::mat4 translation = glm::mat4(
			1.0f, 0.0f, 0.0f, 0.0f,
			0.0f, 1.0f, 0.0f, 0.0f,
			0.0f, 0.0f, 1.0f, 0.0f,
			0.0f, displacement, 0.0f, 1.0f
		);
		view = glm::inverse(translation) * view;
	}
	if (right_mouse_dragging) {
		// Translate on z-axis
		glm::mat4 translation = glm::mat4(
			1.0f, 0.0f, 0.0f, 0.0f,
			0.0f, 1.0f, 0.0f, 0.0f,
			0.0f, 0.0f, 1.0f, 0.0f,
			0.0f, 0.0f, displacement, 1.0f
		);
		view = glm::inverse(translation) * view;
	}
}

void A2::rotateModelBy(float degree) {
	if (left_mouse_dragging) {
		// Rotate around x-axis
		glm::mat4 rotation = glm::mat4(
			1.0f, 0.0f, 0.0f, 0.0f,
			0.0f, cos(degree), sin(degree), 0.0f,
			0.0f, -sin(degree), cos(degree), 0.0f,
			0.0f, 0.0f, 0.0f, 1.0f
		);
		model_transl_rot = model_transl_rot * rotation;
	}
	if (middle_mouse_dragging) {
		// Rotate around y-axis
		glm::mat4 rotation = glm::mat4(
			cos(degree), 0.0f, -sin(degree), 0.0f,
			0.0f, 1.0f, 0.0f, 0.0f,
			sin(degree), 0.0f, cos(degree), 0.0f,
			0.0f, 0.0f, 0.0f, 1.0f
		);
		model_transl_rot = model_transl_rot * rotation;
	}
	if (right_mouse_dragging) {
		// Rotate around z-axis
		glm::mat4 rotation = glm::mat4(
			cos(degree), sin(degree), 0.0f, 0.0f,
			-sin(degree), cos(degree), 0.0f, 0.0f,
			0.0f, 0.0f, 1.0f, 0.0f,
			0.0f, 0.0f, 0.0f, 1.0f
		);
		model_transl_rot = model_transl_rot * rotation;
	}
}

void A2::translateModelBy(float displacement)
{
	if (left_mouse_dragging) {
		// Translate on x-axis
		glm::mat4 translation = glm::mat4(
			1.0f, 0.0f, 0.0f, 0.0f,
			0.0f, 1.0f, 0.0f, 0.0f,
			0.0f, 0.0f, 1.0f, 0.0f,
			displacement, 0.0f, 0.0f, 1.0f
		);
		model_transl_rot = model_transl_rot * translation;
	}
	if (middle_mouse_dragging) {
		// Translate on y-axis
		glm::mat4 translation = glm::mat4(
			1.0f, 0.0f, 0.0f, 0.0f,
			0.0f, 1.0f, 0.0f, 0.0f,
			0.0f, 0.0f, 1.0f, 0.0f,
			0.0f, displacement, 0.0f, 1.0f
		);
		model_transl_rot = model_transl_rot * translation;
	}
	if (right_mouse_dragging) {
		// Translate on z-axis
		glm::mat4 translation = glm::mat4(
			1.0f, 0.0f, 0.0f, 0.0f,
			0.0f, 1.0f, 0.0f, 0.0f,
			0.0f, 0.0f, 1.0f, 0.0f,
			0.0f, 0.0f, displacement, 1.0f
		);
		model_transl_rot = model_transl_rot * translation;
	}
}

void A2::scaleModelBy(float ratio)
{
	if (left_mouse_dragging) {
		// Scale on x-axis
		glm::mat4 scale = glm::mat4(
			ratio, 0.0f, 0.0f, 0.0f,
			0.0f, 0.0f, 0.0f, 0.0f,
			0.0f, 0.0f, 0.0f, 0.0f,
			0.0f, 0.0f, 0.0f, 0.0f
		);
		model_scale = model_scale + scale;
	}
	if (middle_mouse_dragging) {
		// Scale on y-axis
		glm::mat4 scale = glm::mat4(
			0.0f, 0.0f, 0.0f, 0.0f,
			0.0f, ratio, 0.0f, 0.0f,
			0.0f, 0.0f, 0.0f, 0.0f,
			0.0f, 0.0f, 0.0f, 0.0f
		);
		model_scale = model_scale + scale;
	}
	if (right_mouse_dragging) {
		// Scale on z-axis
		glm::mat4 scale = glm::mat4(
			0.0f, 0.0f, 0.0f, 0.0f,
			0.0f, 0.0f, 0.0f, 0.0f,
			0.0f, 0.0f, ratio, 0.0f,
			0.0f, 0.0f, 0.0f, 0.0f
		);
		model_scale = model_scale + scale;
	}
	for (int i = 0; i < 3; i++) {
		model_scale[i][i] = glm::min(model_scale[i][i], MAX_SCALE);
		model_scale[i][i] = glm::max(model_scale[i][i], MIN_SCALE);
	}
}

void A2::modifyPerspective(float deltaX) {
	if (left_mouse_dragging) {
		fov += glm::radians(deltaX / 90);
		fov = glm::min(fov, MAX_FOV);
		fov = glm::max(fov, MIN_FOV);
	}
	if (middle_mouse_dragging) {
		n += deltaX / 20;
	}
	if (right_mouse_dragging) {
		f += deltaX / 20;
	}
	updateProjMatrix();
}

void A2::modifyViewport(double xPos, double yPos) {
	if (left_mouse_dragging) {
		v_x_end = xPos;
		v_x_end = glm::max(double(0), v_x_end);
		v_x_end = glm::min(double(m_framebufferWidth), v_x_end);
		v_y_end = yPos;
		v_y_end = glm::max(double(0), v_y_end);
		v_y_end = glm::min(double(m_framebufferWidth), v_y_end);
		updateViewportMatrix();
	}
}
vector<vec4> A2::clip(vec4 start, vec4 end) {
	vector<vec4> clippedCoords;

	int count = 0;
	float tIn = 0.0f, tOut = 1.0f, tHit;
	float startBC[6], endBC[6];
	int startOutCode = generateOutCode(start);
	int endOutCode = generateOutCode(end);

	if ((startOutCode & endOutCode) != 0) {
		// trivial reject
		return clippedCoords;
	}
	if ((startOutCode | endOutCode) == 0) {
		// trivial accept
		clippedCoords.push_back(start);
		clippedCoords.push_back(end);
		return clippedCoords;
	}

	startBC[0] = start.w + start.x;
	startBC[1] = start.w - start.x;
	startBC[2] = start.w + start.y;
	startBC[3] = start.w - start.y;
	startBC[4] = start.w + start.z;
	startBC[5] = start.w - start.z;

	endBC[0] = end.w + end.x;
	endBC[1] = end.w - end.x;
	endBC[2] = end.w + end.y;
	endBC[3] = end.w - end.y;
	endBC[4] = end.w + end.z;
	endBC[5] = end.w - end.z;

	for (int i = 0; i < 6; i++) {
		if (endBC[i] < 0) {
			// end point is outside the plane i (exiting)
			tHit = startBC[i] / (startBC[i] - endBC[i]); // calculate tHit
			tOut = glm::min(tOut, tHit);
		} else if (startBC[i] < 0) {
			// start point is outside the plain i (entering)
			tHit = startBC[i] / (startBC[i] - endBC[i]);
			tIn = glm::max(tIn, tHit);
		}
		if (tIn > tOut) {
			return clippedCoords;
		}
	}

	if (startOutCode != 0) {
		// start point is outside: tIn has changed, calculate clipped start point
		vec4 clipped = vec4(
			start.x + tIn * (end.x - start.x),
			start.y + tIn * (end.y - start.y),
			start.z + tIn * (end.z - start.z),
			start.w + tIn * (end.w - start.w)
		);
		clippedCoords.push_back(clipped);
	} else {
		clippedCoords.push_back(start);
	}
	if (endOutCode != 0) {
		// end point is outside: tOut has changed, calculate clipped end point
		vec4 clipped = vec4(
			start.x + tOut * (end.x - start.x),
			start.y + tOut * (end.y - start.y),
			start.z + tOut * (end.z - start.z),
			start.w + tOut * (end.w - start.w)
		);
		clippedCoords.push_back(clipped);
	} else {
		clippedCoords.push_back(end);
	}
	return clippedCoords;
}

int A2::generateOutCode(vec4 coord) {
	int outCode = 0;
	float _w = abs(coord.w);
	if (coord.x > _w) outCode |= 1 << 0;
	if (coord.x < -_w) outCode |= 1 << 1;
	if (coord.y > _w) outCode |= 1 << 2;
	if (coord.y < -_w) outCode |= 1 << 3;
	if (coord.z > _w) outCode |= 1 << 4;
	if (coord.z < -_w) outCode |= 1 << 5;
	return outCode;
}

//----------------------------------------------------------------------------------------
/*
 * Called once per frame, after appLogic(), but before the draw() method.
 */
void A2::guiLogic()
{
	static bool firstRun(true);
	if (firstRun) {
		ImGui::SetNextWindowPos(ImVec2(50, 50));
		firstRun = false;
	}

	static bool showDebugWindow(true);
	ImGuiWindowFlags windowFlags(ImGuiWindowFlags_AlwaysAutoResize);
	float opacity(0.5f);

	ImGui::Begin("Properties", &showDebugWindow, ImVec2(100,100), opacity,
			windowFlags);


		// Add more gui elements here here ...


		// Create Button, and check if it was clicked:
		if( ImGui::Button( "Quit Application" ) ) {
			glfwSetWindowShouldClose(m_window, GL_TRUE);
		}
		if (ImGui::Button("Reset")) {
			reset();
		}
		if (ImGui::Button("Rotate View")) {
			curr_mode = ROTATE_VIEW;
		}
		if (ImGui::Button("Translate View")) {
			curr_mode = TRANSLATE_VIEW;
		}
		if (ImGui::Button("Perspective")) {
			curr_mode = PERSPECTIVE;
		}
		if (ImGui::Button("Rotate Model")) {
			curr_mode = ROTATE_MODEL;
		}
		if (ImGui::Button("Translate Model")) {
			curr_mode = TRANSLATE_MODEL;
		}
		if (ImGui::Button("Scale Model")) {
			curr_mode = SCALE_MODEL;
		}
		if (ImGui::Button("Viewport")) {
			curr_mode = VIEWPORT;
		}

		ImGui::Text( "Framerate: %.1f FPS", ImGui::GetIO().Framerate );

	ImGui::End();
}

//----------------------------------------------------------------------------------------
void A2::uploadVertexDataToVbos() {

	//-- Copy vertex position data into VBO, m_vbo_positions:
	{
		glBindBuffer(GL_ARRAY_BUFFER, m_vbo_positions);
		glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vec2) * m_vertexData.numVertices,
				m_vertexData.positions.data());
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		CHECK_GL_ERRORS;
	}

	//-- Copy vertex colour data into VBO, m_vbo_colours:
	{
		glBindBuffer(GL_ARRAY_BUFFER, m_vbo_colours);
		glBufferSubData(GL_ARRAY_BUFFER, 0, sizeof(vec3) * m_vertexData.numVertices,
				m_vertexData.colours.data());
		glBindBuffer(GL_ARRAY_BUFFER, 0);

		CHECK_GL_ERRORS;
	}
}

//----------------------------------------------------------------------------------------
/*
 * Called once per frame, after guiLogic().
 */
void A2::draw()
{
	uploadVertexDataToVbos();

	glBindVertexArray(m_vao);

	m_shader.enable();
		glDrawArrays(GL_LINES, 0, m_vertexData.numVertices);
	m_shader.disable();

	// Restore defaults
	glBindVertexArray(0);

	CHECK_GL_ERRORS;
}

//----------------------------------------------------------------------------------------
/*
 * Called once, after program is signaled to terminate.
 */
void A2::cleanup()
{

}

//----------------------------------------------------------------------------------------
/*
 * Event handler.  Handles cursor entering the window area events.
 */
bool A2::cursorEnterWindowEvent (
		int entered
) {
	bool eventHandled(false);

	// Fill in with event handling code...

	return eventHandled;
}

//----------------------------------------------------------------------------------------
/*
 * Event handler.  Handles mouse cursor movement events.
 */
bool A2::mouseMoveEvent (
		double xPos,
		double yPos
) {
	bool eventHandled(false);

	// Fill in with event handling code...
	if (!ImGui::IsMouseHoveringAnyWindow()) {
		double deltaX = xPos - prev_x_pos;
		//double deltaY = yPos - prev_y_pos;
		switch (curr_mode) {
			case ROTATE_VIEW:
				rotateViewBy(glm::radians(float(deltaX / 3)));
				break;
			case TRANSLATE_VIEW:
				translateViewBy(float(deltaX / 20));
				break;
			case PERSPECTIVE:
				modifyPerspective(float(deltaX));
				break;
			case ROTATE_MODEL:
				rotateModelBy(glm::radians(float(deltaX / 3)));
				break;
			case TRANSLATE_MODEL:
				translateModelBy(float(deltaX / 20));
				break;
			case SCALE_MODEL:
				scaleModelBy(float(deltaX / 20));
				break;
			case VIEWPORT:
				modifyViewport(xPos, yPos);
				break;
		}
		
		prev_x_pos = xPos;
		eventHandled = true;
	}

	return eventHandled;
}

//----------------------------------------------------------------------------------------
/*
 * Event handler.  Handles mouse button events.
 */
bool A2::mouseButtonInputEvent (
		int button,
		int actions,
		int mods
) {
	bool eventHandled(false);

	// Fill in with event handling code...
	if (!ImGui::IsMouseHoveringAnyWindow()) {
		if (actions == GLFW_PRESS) {
			if (button == GLFW_MOUSE_BUTTON_LEFT) {
				left_mouse_dragging = true;
				if (curr_mode == VIEWPORT) {
					glfwGetCursorPos(m_window, &v_x_start, &v_y_start);
				}
			}
			else if (button == GLFW_MOUSE_BUTTON_MIDDLE) middle_mouse_dragging = true;
			else if (button == GLFW_MOUSE_BUTTON_RIGHT) right_mouse_dragging = true;
		}
		else if (actions == GLFW_RELEASE) {
			if (button == GLFW_MOUSE_BUTTON_LEFT) left_mouse_dragging = false;
			else if (button == GLFW_MOUSE_BUTTON_MIDDLE) middle_mouse_dragging = false;
			else if (button == GLFW_MOUSE_BUTTON_RIGHT) right_mouse_dragging = false;
		}
		eventHandled = true;
	}

	return eventHandled;
}

//----------------------------------------------------------------------------------------
/*
 * Event handler.  Handles mouse scroll wheel events.
 */
bool A2::mouseScrollEvent (
		double xOffSet,
		double yOffSet
) {
	bool eventHandled(false);

	// Fill in with event handling code...

	return eventHandled;
}

//----------------------------------------------------------------------------------------
/*
 * Event handler.  Handles window resize events.
 */
bool A2::windowResizeEvent (
		int width,
		int height
) {
	bool eventHandled(false);

	// Fill in with event handling code...

	return eventHandled;
}

//----------------------------------------------------------------------------------------
/*
 * Event handler.  Handles key input events.
 */
bool A2::keyInputEvent (
		int key,
		int action,
		int mods
) {
	bool eventHandled(false);

	// Fill in with event handling code...
	if (action == GLFW_PRESS) {
		// Respond to some key events.
		switch (key) {
		case GLFW_KEY_A:
			reset();
			break;
		case GLFW_KEY_Q:
			glfwSetWindowShouldClose(m_window, GL_TRUE);
			break;
		case GLFW_KEY_O:
			curr_mode = ROTATE_VIEW;
			break;
		case GLFW_KEY_N:
			curr_mode = TRANSLATE_VIEW;
			break;
		case GLFW_KEY_P:
			curr_mode = PERSPECTIVE;
			break;
		case GLFW_KEY_R:
			curr_mode = ROTATE_MODEL;
			break;
		case GLFW_KEY_T:
			curr_mode = TRANSLATE_MODEL;
			break;
		case GLFW_KEY_S:
			curr_mode = SCALE_MODEL;
			break;
		case GLFW_KEY_V:
			curr_mode = VIEWPORT;
			break;
		}
	}

	return eventHandled;
}
