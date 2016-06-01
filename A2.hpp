#pragma once

#include "cs488-framework/CS488Window.hpp"
#include "cs488-framework/OpenGLImport.hpp"
#include "cs488-framework/ShaderProgram.hpp"

#include <glm/glm.hpp>

#include <vector>

// Set a global maximum number of vertices in order to pre-allocate VBO data
// in one shot, rather than reallocating each frame.
const GLsizei kMaxVertices = 1000;


enum Mode {
	ROTATE_VIEW,
	TRANSLATE_VIEW,
	PERSPECTIVE,
	ROTATE_MODEL,
	TRANSLATE_MODEL,
	SCALE_MODEL,
	VIEWPORT
};

enum Option {
	X,
	Y,
	Z
};


// Convenience class for storing vertex data in CPU memory.
// Data should be copied over to GPU memory via VBO storage before rendering.
class VertexData {
public:
	VertexData();

	std::vector<glm::vec2> positions;
	std::vector<glm::vec3> colours;
	GLuint index;
	GLsizei numVertices;
};


class A2 : public CS488Window {
public:
	A2();
	virtual ~A2();

protected:
	virtual void init() override;
	virtual void appLogic() override;
	virtual void guiLogic() override;
	virtual void draw() override;
	virtual void cleanup() override;

	virtual bool cursorEnterWindowEvent(int entered) override;
	virtual bool mouseMoveEvent(double xPos, double yPos) override;
	virtual bool mouseButtonInputEvent(int button, int actions, int mods) override;
	virtual bool mouseScrollEvent(double xOffSet, double yOffSet) override;
	virtual bool windowResizeEvent(int width, int height) override;
	virtual bool keyInputEvent(int key, int action, int mods) override;

	void createShaderProgram();
	void enableVertexAttribIndices();
	void generateVertexBuffers();
	void mapVboDataToVertexAttributeLocation();
	void uploadVertexDataToVbos();

	void initLineData();

	void setLineColour(const glm::vec3 & colour);

	void drawLine (
			const glm::vec2 & v0,
			const glm::vec2 & v1
	);

	void reset();
	void rotateViewBy(float degree);
	void translateViewBy(float displacement);
	void rotateModelBy(float degree);
	void translateModelBy(float displacement);
	void scaleModelBy(float ratio);
	void updateProjMatrix();
	void updateViewportMatrix();
	void modifyPerspective(float deltaX);
	void modifyViewport(double xPos, double yPos);
	std::vector<glm::vec4> clip(glm::vec4 start, glm::vec4 end);
	int generateOutCode(glm::vec4 coord);

	ShaderProgram m_shader;

	GLuint m_vao;            // Vertex Array Object
	GLuint m_vbo_positions;  // Vertex Buffer Object
	GLuint m_vbo_colours;    // Vertex Buffer Object

	VertexData m_vertexData;

	glm::vec3 m_currentLineColour;

	glm::mat4 proj;
	glm::mat4 view;
	glm::mat4 model_scale;
	glm::mat4 model_transl_rot;
	glm::mat3 viewport;

	bool left_mouse_dragging;
	bool middle_mouse_dragging;
	bool right_mouse_dragging;

	float fov;
	float f;
	float n;
	double prev_x_pos;
	double prev_y_pos;

	double v_x_start;
	double v_x_end;
	double v_y_start;
	double v_y_end;
	
	Mode curr_mode;
};

